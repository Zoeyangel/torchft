# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import threading
from datetime import timedelta
from typing import Callable, List, Optional, Sequence, Tuple
from unittest import TestCase
from unittest.mock import create_autospec, MagicMock, patch

import torch
from torch.distributed import TCPStore
from torchft._torchft import QuorumResult
from torchft.elastic import (
    DatasetIndexMapper,
    ElasticBatchPlanner,
    local_sample_indices,
    replica_slot,
)
from torchft.manager import (
    Manager,
    MANAGER_ADDR_KEY,
    QuorumView,
    REPLICA_ID_KEY,
    WorldSizeMode,
)
from torchft.process_group import ManagedProcessGroup, ProcessGroup


def quorum_ids(slots: Sequence[int]) -> List[str]:
    """Quorum replica ids for ``slots``: replica_id plus a per-process UUID."""
    return [f"replica_{slot}:uuid{slot}" for slot in slots]


def lighthouse_order(slots: Sequence[int]) -> List[int]:
    """The participating_rank order: the lighthouse sorts by replica_id string."""
    return sorted(slots, key=lambda slot: quorum_ids([slot])[0])


class FakeManager:
    """The subset of Manager the planner uses, with a settable quorum.

    wait_plan() runs the registered plan function on the current quorum, as
    the quorum thread does at the membership point.
    """

    def __init__(
        self, slot: Optional[int] = 0, min_replica_size: int = 1
    ) -> None:
        self._replica_id = None if slot is None else f"replica_{slot}"
        self._min = min_replica_size
        self.qid = -1
        self.ids: List[str] = []
        self.idx: Optional[int] = None
        self.step = 0
        self.plan_fn: Optional[Callable[[QuorumView], object]] = None

    def set(
        self, qid: int, slots: Sequence[int], idx: Optional[int], step: int = 0
    ) -> None:
        """``slots`` in participating_rank order; ``idx`` is this group's rank."""
        self.qid, self.ids, self.idx, self.step = qid, quorum_ids(slots), idx, step

    def replica_id(self) -> Optional[str]:
        return self._replica_id

    def min_replica_size(self) -> int:
        return self._min

    def set_plan_fn(self, fn: Optional[Callable[[QuorumView], object]]) -> None:
        self.plan_fn = fn

    def wait_plan(self) -> object:
        assert self.plan_fn is not None
        return self.plan_fn(
            QuorumView(
                quorum_id=self.qid,
                step=self.step,
                num_participants=len(self.ids),
                participating_rank=self.idx,
                participating_replica_ids=tuple(self.ids),
            )
        )


def make_planner(manager: object, N: int = 4, m: int = 3) -> ElasticBatchPlanner:
    # pyrefly: ignore [bad-argument-type]
    return ElasticBatchPlanner(manager, N, m)


# (base units, tail units) of one replica group: what the planner hands out
Assignment = Tuple[Tuple[int, ...], Tuple[int, ...]]


def assignments_for_quorum(
    present: Sequence[int],
    step: int,
    N: int = 4,
    m: int = 3,
) -> List[Assignment]:
    """The assignment of every participant of a quorum of ``present`` slots."""
    order = lighthouse_order(present)
    assignments = []
    for idx, slot in enumerate(order):
        manager = FakeManager(slot)
        manager.set(qid=7, slots=order, idx=idx, step=step)
        planner = make_planner(manager, N, m)
        assignments.append((planner.base_units(step), planner.plan()))
    return assignments


class TestSharding(TestCase):
    def test_partition_exact(self) -> None:
        # every non-empty membership, even and uneven tails
        for N, m in [(2, 1), (3, 2), (4, 3), (5, 1), (11, 2)]:
            U = N * m
            subsets = [
                s
                for size in range(1, N + 1)
                for s in itertools.combinations(range(N), size)
            ]
            # N=11 has 2047 subsets; a sample with gaps and two-digit slots
            # is enough
            for present in subsets[:: 1 if N < 11 else 37]:
                for step in (0, 1, 13):
                    assignments = assignments_for_quorum(present, step, N=N, m=m)
                    msg = f"N={N} m={m} present={present} step={step}"
                    units = [u for base, tail in assignments for u in base + tail]
                    self.assertEqual(
                        sorted(units), list(range(step * U, (step + 1) * U)), msg
                    )
                    gas = [len(base) + len(tail) for base, tail in assignments]
                    self.assertEqual(sum(gas), U, msg)
                    # nobody gives up its base, the tail is split evenly
                    self.assertGreaterEqual(min(gas), m, msg)
                    self.assertLessEqual(max(gas) - min(gas), 1, msg)

    def test_base_needs_no_quorum(self) -> None:
        # a manager that never ran a quorum still knows this group's units:
        # base_units is arithmetic on the step and the slot
        for slot in range(4):
            planner = make_planner(FakeManager(slot))
            self.assertEqual(
                planner.base_units(5), tuple(range(60 + 3 * slot, 63 + 3 * slot))
            )

    def test_full_quorum_has_no_tail(self) -> None:
        for base, tail in assignments_for_quorum((0, 1, 2, 3), step=2):
            self.assertEqual(tail, ())
            self.assertEqual(len(base), 3)

    def test_gap_in_slots(self) -> None:
        # slots 0, 1, 3 present: slot 2's units 6..8 go one to each
        assignments = assignments_for_quorum((0, 1, 3), step=0)
        self.assertEqual([tail for _, tail in assignments], [(6,), (7,), (8,)])

    def test_tail_follows_participating_rank(self) -> None:
        # replica_10 sorts before replica_2: participating_rank 0 is slot 10,
        # and it takes the first share of the tail
        order = lighthouse_order((2, 10))
        self.assertEqual(order, [10, 2])
        assignments = assignments_for_quorum((2, 10), step=0, N=11, m=1)
        # 9 absent units, 5 then 4
        self.assertEqual([len(tail) for _, tail in assignments], [5, 4])
        self.assertEqual(assignments[0][0], (10,))

    def test_non_participant(self) -> None:
        m = FakeManager(slot=2)
        m.set(qid=3, slots=[0, 1], idx=None, step=5)
        planner = make_planner(m)
        # it gets no tail, but it still runs its base: the Manager zeroes the
        # gradient of a group outside the quorum
        self.assertEqual(planner.plan(), ())
        self.assertEqual(planner.base_units(5), (66, 67, 68))
        self.assertEqual(planner.U, 12)


class TestValidation(TestCase):
    def test_construction(self) -> None:
        for kwargs in [{"m": 0}, {"N": 0}]:
            with self.assertRaises(ValueError, msg=kwargs):
                make_planner(FakeManager(), **kwargs)
        # N - 1 < min_replica_size
        with self.assertRaisesRegex(ValueError, "min_replica_size"):
            make_planner(FakeManager(min_replica_size=3), N=3)
        # no replica_id, no trailing digits, slot out of range
        with self.assertRaisesRegex(ValueError, "replica_id"):
            make_planner(FakeManager(slot=None))
        bad = FakeManager()
        bad._replica_id = "replica"
        with self.assertRaisesRegex(ValueError, "slot number"):
            make_planner(bad)
        with self.assertRaisesRegex(ValueError, "outside"):
            make_planner(FakeManager(slot=4), N=4)

    def test_sizes(self) -> None:
        self.assertEqual(make_planner(FakeManager(), N=4, m=3).U, 12)

    def test_replica_slot(self) -> None:
        self.assertEqual(replica_slot("fsdpturbo_replica_12"), 12)
        self.assertEqual(replica_slot("3:0a1b-22"), 3)
        with self.assertRaises(ValueError):
            replica_slot("replica:7")

    def test_no_quorum(self) -> None:
        planner = make_planner(FakeManager())
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            planner.plan()

    def test_ids_must_match_participants(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        assert m.plan_fn is not None
        view = QuorumView(quorum_id=1, step=0, num_participants=2, participating_rank=0)
        with self.assertRaisesRegex(RuntimeError, "max_replica_ids"):
            planner.plan_from_view(view)

    def test_slots_must_be_distinct_and_in_range(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        m.set(qid=1, slots=[0, 0], idx=0)
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            planner.plan()
        m.set(qid=2, slots=[0, 4], idx=0)
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            planner.plan()

    def test_own_slot_must_match_rank(self) -> None:
        m = FakeManager(slot=1)
        planner = make_planner(m)
        m.set(qid=1, slots=[0, 1], idx=0)
        with self.assertRaisesRegex(RuntimeError, "not this group's slot"):
            planner.plan()

    def test_below_min_replica_size_does_not_raise(self) -> None:
        # P < min: still planned; should_commit blocks the commit
        m = FakeManager(min_replica_size=2)
        planner = make_planner(m)
        m.set(qid=1, slots=[0], idx=0)
        self.assertEqual(len(planner.base_units(0) + planner.plan()), 12)


class TestReplan(TestCase):
    def test_registers_with_manager(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        self.assertEqual(m.plan_fn, planner.plan_from_view)

    def test_plan_is_one_quorum(self) -> None:
        m = FakeManager(slot=1)
        planner = make_planner(m)
        m.set(qid=1, slots=[0, 1], idx=1, step=3)
        planner.plan()
        m.set(qid=2, slots=[0, 1, 2], idx=1, step=4)
        plan = planner.plan()
        self.assertEqual(plan, (4 * 12 + 10,))

    def test_uuid_or_quorum_id_change_does_not_reassign(self) -> None:
        # a replacement group with a new UUID in the same slot, or a bare
        # quorum bump, cannot change the assignment
        m = FakeManager()
        planner = make_planner(m)
        m.set(qid=1, slots=[0, 2], idx=0)
        first = planner.plan()
        m.ids = ["replica_0:other", "replica_2:new"]
        m.qid = 2
        with patch.object(planner, "_assign", wraps=planner._assign) as assign:
            plan = planner.plan()
        assign.assert_not_called()
        self.assertEqual(plan, first)

    def test_membership_change_reassigns(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        m.set(qid=1, slots=[0, 1, 2, 3], idx=0)
        self.assertEqual(planner.plan(), ())
        # slot 3 leaves, slot 2's group takes its place in the order
        m.set(qid=2, slots=[0, 1, 2], idx=0)
        self.assertEqual(planner.plan(), (9,))
        # same P, different slots: a different absent slot
        m.set(qid=3, slots=[0, 1, 3], idx=0)
        self.assertEqual(planner.plan(), (6,))

    def test_promoted_from_spare(self) -> None:
        m = FakeManager(slot=1)
        planner = make_planner(m)
        m.set(qid=5, slots=[0, 2], idx=None, step=3)
        self.assertEqual(planner.plan(), ())
        m.set(qid=5, slots=[0, 1], idx=1, step=4)
        self.assertEqual(planner.plan(), (4 * 12 + 9, 4 * 12 + 10, 4 * 12 + 11))

    def test_retry_reads_same_units(self) -> None:
        # should_commit False -> step does not advance -> identical units
        m = FakeManager(slot=1)
        planner = make_planner(m)
        m.set(qid=1, slots=[0, 1], idx=1, step=4)
        a = planner.plan()
        m.qid = 2
        b = planner.plan()
        self.assertEqual(a, b)

    def test_cached_matches_fresh(self) -> None:
        m = FakeManager(slot=1)
        planner = make_planner(m)
        memberships = [[0, 1], [1], [0, 1, 2, 3], [1, 3], [0, 2], [0, 1]]
        for step, slots in enumerate(memberships, start=1):
            idx = slots.index(1) if 1 in slots else None
            m.set(qid=step // 2, slots=slots, idx=idx, step=step)
            cached = planner.plan()
            # a separate manager: a second planner would replace the first
            # one's registration
            fresh_m = FakeManager(slot=1)
            fresh_m.set(qid=step // 2, slots=slots, idx=idx, step=step)
            self.assertEqual(cached, make_planner(fresh_m).plan())


class TestLocalSampleIndices(TestCase):
    def test_split_across_ranks(self) -> None:
        micro, D = 2, 3
        units = (4, 9, 10)
        parts = [local_sample_indices(units, micro, r, D) for r in range(D)]
        for part in parts:
            self.assertEqual(len(part), len(units) * micro)
        # micro-batch i of every rank together is unit i
        for i, unit in enumerate(units):
            batch = sorted(x for part in parts for x in part[i * micro : (i + 1) * micro])
            self.assertEqual(batch, list(range(unit * micro * D, (unit + 1) * micro * D)))

    def test_no_units(self) -> None:
        self.assertEqual(local_sample_indices((), 2, 0, 2), [])


class TestDatasetIndexMapper(TestCase):
    def test_epoch_is_permutation(self) -> None:
        mapper = DatasetIndexMapper(dataset_len=17, seed=3)
        for epoch in range(3):
            got = [mapper(epoch * 17 + i) for i in range(17)]
            self.assertEqual(sorted(got), list(range(17)))

    def test_epochs_differ_and_cross_boundary(self) -> None:
        mapper = DatasetIndexMapper(dataset_len=50, seed=0)
        e0 = [mapper(i) for i in range(50)]
        e1 = [mapper(50 + i) for i in range(50)]
        self.assertNotEqual(e0, e1)
        # a window spanning the boundary reads the tail of e0 then the head of e1
        self.assertEqual([mapper(i) for i in range(45, 55)], e0[45:] + e1[:5])

    def test_deterministic_across_instances(self) -> None:
        a = DatasetIndexMapper(dataset_len=1000, seed=11)
        b = DatasetIndexMapper(dataset_len=1000, seed=11)
        # different access orders, same result
        idx = [5, 1999, 3, 1000, 2500]
        self.assertEqual([a(i) for i in idx], [b(i) for i in reversed(idx)][::-1])

    def test_no_shuffle(self) -> None:
        mapper = DatasetIndexMapper(dataset_len=10, shuffle=False)
        self.assertEqual([mapper(i) for i in range(8, 13)], [8, 9, 0, 1, 2])


def mock_should_commit(
    rank: int, step: int, should_commit: bool, timeout: timedelta
) -> bool:
    return should_commit


# this replica group's slot in TestWithManager
OWN_SLOT = 1


class TestWithManager(TestCase):
    """Runs the planner against the real Manager with a mocked quorum client.

    The planner defaults give U = 12: N = 4 slots of m = 3, rb = 4.
    """

    store: TCPStore  # pyre-fixme[13]: never initialized

    def _create_manager(
        self,
        min_replica_size: int = 1,
        world_size_mode: WorldSizeMode = WorldSizeMode.DYNAMIC,
        use_async_quorum: bool = True,
    ) -> Manager:
        pg = create_autospec(ProcessGroup)
        # pyrefly: ignore [missing-attribute]
        pg.errored.return_value = None
        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store.set(MANAGER_ADDR_KEY, "dummy")
        self.store.set(REPLICA_ID_KEY, "dummy_id")
        with patch(
            "os.environ",
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": self.store.port,
                "RANK": "1",
                "WORLD_SIZE": "2",
            },
        ):
            manager = Manager(
                pg=pg,
                min_replica_size=min_replica_size,
                load_state_dict=MagicMock(),
                state_dict=lambda: {},
                replica_id=f"replica_{OWN_SLOT}",
                use_async_quorum=use_async_quorum,
                world_size_mode=world_size_mode,
                init_sync=False,
            )
        self.addCleanup(manager.shutdown, wait=False)
        return manager

    def _quorum(
        self,
        quorum_id: int,
        max_replica_rank: Optional[int],
        max_world_size: int,
        step: int = 0,
    ) -> QuorumResult:
        """Every participant at max_step; this group at max_replica_rank."""
        others = [s for s in range(16) if s != OWN_SLOT]
        if max_replica_rank is None:
            slots = others[:max_world_size]
        else:
            slots = others[: max_world_size - 1]
            slots.insert(max_replica_rank, OWN_SLOT)
        quorum = QuorumResult()
        quorum.quorum_id = quorum_id
        quorum.replica_rank = 0 if max_replica_rank is None else max_replica_rank
        quorum.replica_world_size = max_world_size
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = step
        quorum.max_replica_rank = max_replica_rank
        quorum.max_world_size = max_world_size
        quorum.replica_ids = quorum_ids(slots)
        quorum.max_replica_ids = quorum_ids(slots)
        quorum.heal = False
        return quorum

    def _healing_quorum(self, manager: Manager, client_mock: MagicMock) -> None:
        # this replica group (rank 1 of 2) is at step 0 and heals to step 20
        # from rank 0, the only one at max_step
        quorum = self._quorum(123, None, 1, step=20)
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.replica_ids = quorum_ids([0, OWN_SLOT])
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.heal = True
        client_mock()._quorum.return_value = quorum
        # stage a checkpoint for the transport to serve back
        manager._checkpoint_transport.send_checkpoint(
            dst_ranks=[],
            step=quorum.max_step,
            state_dict=manager._manager_state_dict(),
            timeout=timedelta(seconds=10),
        )
        client_mock()._checkpoint_metadata.return_value = (
            manager._checkpoint_transport.metadata()
        )

    def _record_plan_views(
        self, manager: Manager
    ) -> Tuple[List[int], List[QuorumView]]:
        # the Manager step and the view at the moment the plan fn runs
        plan_fn = manager._plan_fn
        assert plan_fn is not None
        steps: List[int] = []
        views: List[QuorumView] = []

        def recording(view: QuorumView) -> object:
            steps.append(manager._step)
            views.append(view)
            return plan_fn(view)

        manager.set_plan_fn(recording)
        return steps, views

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_heal_plans_after_healing(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager)
        steps, views = self._record_plan_views(manager)
        self._healing_quorum(manager, client_mock)

        stale_base = planner.base_units(manager.current_step())
        manager.start_quorum()
        plan = planner.plan()
        # healed before the plan: it trains step 20 with every replica group,
        # so the base read before the quorum is stale
        self.assertEqual(steps, [20])
        self.assertEqual(
            (views[0].step, views[0].num_participants, views[0].participating_rank),
            (20, 2, 1),
        )
        self.assertEqual(manager.current_step(), 20)
        self.assertEqual(stale_base, (3, 4, 5))
        self.assertEqual(planner.base_units(20), (243, 244, 245))
        # slots 2 and 3 are absent: units 6..11, the second half is ours
        self.assertEqual(plan, (249, 250, 251))
        self.assertEqual(ManagedProcessGroup(manager).size(), 2)
        self.assertTrue(manager.is_participating())
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())
        self.assertEqual(manager.current_step(), 21)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_async_heal_plans_before_healing(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=True)
        planner = make_planner(manager)
        steps, views = self._record_plan_views(manager)
        self._healing_quorum(manager, client_mock)

        manager.start_quorum()
        plan = planner.plan()
        # outside the max-step view while healing: its base only, zeroed
        self.assertEqual(
            (views[0].step, views[0].num_participants, views[0].participating_rank),
            (0, 1, None),
        )
        self.assertEqual(plan, ())
        self.assertEqual(planner.base_units(0), (3, 4, 5))
        manager.wait_quorum()
        self.assertEqual(steps, [0])
        self.assertEqual(manager.current_step(), 20)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_reconfigure_error_still_plans(self, client_mock: MagicMock) -> None:
        # _async_quorum only reports a reconfigure error; wait_plan() must
        # still return, and the step does not commit
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager)
        _, views = self._record_plan_views(manager)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = RuntimeError("configure failed")
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)

        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual(
            (views[0].step, views[0].num_participants, views[0].participating_rank),
            (0, 3, 1),
        )
        self.assertEqual(plan, (10,))
        self.assertIsNotNone(manager.errored())
        self.assertFalse(manager.should_commit())

    def _duplicate_slot_quorum(self) -> QuorumResult:
        quorum = self._quorum(1, 0, 2)
        quorum.replica_ids = quorum.max_replica_ids = ["replica_1:a", "replica_1:b"]
        return quorum

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_plan_error_does_not_fail_the_quorum(
        self, client_mock: MagicMock
    ) -> None:
        manager = self._create_manager(use_async_quorum=False)
        planner = make_planner(manager)
        client_mock()._quorum.return_value = self._duplicate_slot_quorum()
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            planner.plan()
        self.assertIsNone(manager.errored())
        self.assertEqual(ManagedProcessGroup(manager).size(), 2)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_accessors(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(min_replica_size=3)
        self.assertEqual(manager.min_replica_size(), 3)
        self.assertEqual(manager.replica_id(), "replica_1")
        client_mock()._quorum.return_value = self._quorum(42, 0, 4)
        manager.start_quorum()
        manager.wait_quorum()
        self.assertEqual(manager.participating_rank(), 0)
        self.assertEqual(manager.num_participants(), 4)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_before_start_quorum(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            planner.plan()

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_reads_the_new_quorum(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager)
        _, views = self._record_plan_views(manager)

        # slots 0, 1, 2: slot 3's units 9..11, one each, ours is 10
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual(views[0].quorum_id, 1)
        self.assertEqual(plan, (10,))
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())

        # next quorum: slots 0, 1 at step 1, ours is the second half of 6..11
        client_mock()._quorum.return_value = self._quorum(2, 1, 2, step=1)
        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual((views[1].quorum_id, views[1].step), (2, 1))
        self.assertEqual(plan, (21, 22, 23))

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_fixed_with_spares(self, client_mock: MagicMock) -> None:
        # P is clamped to min_replica_size and ranks >= min become spares; a
        # spare's slot is read by the active groups
        manager = self._create_manager(
            min_replica_size=2, world_size_mode=WorldSizeMode.FIXED_WITH_SPARES
        )
        planner = make_planner(manager)
        _, views = self._record_plan_views(manager)

        client_mock()._quorum.return_value = self._quorum(1, 2, 3)
        manager.start_quorum()
        spare = planner.plan()
        self.assertEqual(
            (views[0].num_participants, views[0].participating_rank), (2, None)
        )
        self.assertEqual(spare, ())

        # a lower-ranked replica leaves: same P after the clamp, idx 2 -> 1
        client_mock()._quorum.return_value = self._quorum(2, 1, 2)
        manager.start_quorum()
        promoted = planner.plan()
        self.assertEqual(
            (views[1].num_participants, views[1].participating_rank), (2, 1)
        )
        self.assertEqual(promoted, (9, 10, 11))

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_does_not_wait_for_reconfigure(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        _, views = self._record_plan_views(manager)
        release = threading.Event()
        self.addCleanup(release.set)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = lambda *_: release.wait(10)

        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        planner.plan()
        self.assertEqual(
            (views[0].quorum_id, views[0].num_participants, views[0].participating_rank),
            (1, 3, 1),
        )
        # FSDP2 reads the replicate PG size on every micro-batch
        self.assertEqual(ManagedProcessGroup(manager).size(), 3)
        assert manager._quorum_future is not None
        self.assertFalse(manager._quorum_future.done())

        release.set()
        manager.wait_quorum()
        self.assertEqual(manager.num_participants(), 3)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_error_does_not_fail_the_quorum(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        client_mock()._quorum.return_value = self._duplicate_slot_quorum()
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            planner.plan()
        manager.wait_quorum()
        self.assertIsNone(manager.errored())
        self.assertEqual(ManagedProcessGroup(manager).size(), 2)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_error_reaches_plan(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        client_mock()._quorum.side_effect = RuntimeError("quorum rpc failed")
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "quorum rpc failed"):
            planner.plan()
        with self.assertRaisesRegex(RuntimeError, "quorum rpc failed"):
            ManagedProcessGroup(manager).size()

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_size_without_planner(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            manager.wait_plan()
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        self.assertEqual(ManagedProcessGroup(manager).size(), 3)
        assert manager._quorum_future is not None
        self.assertTrue(manager._quorum_future.done())

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_set_plan_fn_while_quorum_in_progress(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        release = threading.Event()
        self.addCleanup(release.set)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = lambda *_: release.wait(10)

        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        # the plan is already out, but the round is still in reconfigure: a
        # change here would strand the next round's plan, so it is refused
        self.assertEqual(planner.plan(), (10,))
        with self.assertRaisesRegex(RuntimeError, "in progress"):
            manager.set_plan_fn(None)

        release.set()
        manager.wait_quorum()
        # between rounds it can be unregistered and the next round has no plan
        manager.set_plan_fn(None)
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            manager.wait_plan()
        self.assertEqual(ManagedProcessGroup(manager).size(), 3)
