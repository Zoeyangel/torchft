# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import threading
from datetime import timedelta
from typing import List, Optional, Sequence, Tuple
from unittest import TestCase
from unittest.mock import create_autospec, MagicMock, patch

import torch
from torch.distributed import TCPStore
from torchft._torchft import QuorumResult
from torchft.elastic import (
    ElasticBatchPlanner,
    replica_slot,
)
from torchft.manager import (
    Manager,
    MANAGER_ADDR_KEY,
    REPLICA_ID_KEY,
    WorldSizeMode,
)
from torchft.process_group import ProcessGroup


def quorum_ids(slots: Sequence[int]) -> List[str]:
    """Quorum replica ids for ``slots``: replica_id plus a per-process UUID."""
    return [f"replica_{slot}:uuid{slot}" for slot in slots]


def lighthouse_order(slots: Sequence[int]) -> List[int]:
    return sorted(slots, key=lambda slot: quorum_ids([slot])[0])


class FakeManager:
    """The subset of Manager the planner uses, with a settable quorum.

    wait_membership() hands out the current quorum's participants, as the real
    Manager does at the membership point. The constructor value is kept under
    the Manager's own attribute name so make_planner can read it from either
    this double or a real Manager.
    """

    def __init__(self, slot: Optional[int] = 0) -> None:
        self._replica_id = None if slot is None else f"replica_{slot}"
        self.ids: List[str] = []
        self.step = 0

    def set(self, slots: Sequence[int], step: int = 0) -> None:
        """``slots`` in the order the Manager reports them."""
        self.ids, self.step = quorum_ids(slots), step

    def wait_membership(self) -> Tuple[str, ...]:
        return tuple(self.ids)

    def current_step(self) -> int:
        return self.step


def make_planner(manager: object, N: int = 4, m: int = 3) -> ElasticBatchPlanner:
    return ElasticBatchPlanner(
        manager,  # pyrefly: ignore [bad-argument-type]
        manager._replica_id,  # pyrefly: ignore [missing-attribute]
        N,
        m,
    )


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
    for slot in present:
        manager = FakeManager(slot)
        # the planner reads its position off the order the Manager reports
        manager.set(slots=order, step=step)
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

    def test_gap_in_slots(self) -> None:
        # slots 0, 1, 3 present: slot 2's units 6..8 go one to each
        assignments = assignments_for_quorum((0, 1, 3), step=0)
        self.assertEqual([tail for _, tail in assignments], [(6,), (7,), (8,)])

    def test_tail_follows_the_reported_order(self) -> None:
        # replica_10 sorts before replica_2, and the Manager puts the max_step
        # group in front: replica_10 takes the first share of the tail
        order = lighthouse_order((2, 10))
        self.assertEqual(order, [10, 2])
        assignments = assignments_for_quorum((2, 10), step=0, N=11, m=1)
        # 9 absent units, by reported position
        by_slot = dict(zip((2, 10), assignments))
        self.assertEqual([len(by_slot[slot][1]) for slot in order], [5, 4])
        self.assertEqual(by_slot[10][0], (10,))

    def test_non_participant(self) -> None:
        m = FakeManager(slot=2)
        m.set(slots=[0, 1], step=5)
        planner = make_planner(m)
        # it gets no tail, but it still runs its base: the Manager zeroes the
        # gradient of a group outside the quorum
        self.assertEqual(planner.plan(), ())
        self.assertEqual(planner.base_units(5), (66, 67, 68))


class TestValidation(TestCase):
    def test_replica_slot(self) -> None:
        self.assertEqual(replica_slot("fsdpturbo_replica_12"), 12)
        self.assertEqual(replica_slot("3:0a1b-22"), 3)
        with self.assertRaises(ValueError):
            replica_slot("replica:7")

    def test_slots_must_be_distinct_and_in_range(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        m.set(slots=[0, 0])
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            planner.plan()
        m.set(slots=[0, 4])
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            planner.plan()


class TestReplan(TestCase):
    def test_plan_is_one_quorum(self) -> None:
        m = FakeManager(slot=1)
        planner = make_planner(m)
        m.set(slots=[0, 1], step=3)
        planner.plan()
        m.set(slots=[0, 1, 2], step=4)
        plan = planner.plan()
        self.assertEqual(plan, (4 * 12 + 10,))

    def test_uuid_change_does_not_reassign(self) -> None:
        # a replacement group with a new UUID in the same slot cannot change
        # the assignment
        m = FakeManager()
        planner = make_planner(m)
        m.set(slots=[0, 2])
        first = planner.plan()
        m.ids = ["replica_0:other", "replica_2:new"]
        self.assertEqual(planner.plan(), first)

    def test_membership_change_reassigns(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        m.set(slots=[0, 1, 2, 3])
        self.assertEqual(planner.plan(), ())
        # slot 3 leaves, slot 2's group takes its place in the order
        m.set(slots=[0, 1, 2])
        self.assertEqual(planner.plan(), (9,))
        # same P, different slots: a different absent slot
        m.set(slots=[0, 1, 3])
        self.assertEqual(planner.plan(), (6,))

    def test_promoted_from_spare(self) -> None:
        m = FakeManager(slot=1)
        planner = make_planner(m)
        m.set(slots=[0, 2], step=3)
        self.assertEqual(planner.plan(), ())
        m.set(slots=[0, 1], step=4)
        self.assertEqual(planner.plan(), (4 * 12 + 9, 4 * 12 + 10, 4 * 12 + 11))


def mock_should_commit(
    rank: int, step: int, should_commit: bool, timeout: timedelta
) -> bool:
    return should_commit


# this replica group's slot in TestWithManager
OWN_SLOT = 1


class TestWithManager(TestCase):
    """Runs the planner against the real Manager with a mocked quorum client.

    The planner defaults give U = 12: N = 4 slots of m = 3.
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
        quorum.replica_ids = quorum_ids(lighthouse_order(slots))
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

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_heal_plans_after_healing(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager)
        self._healing_quorum(manager, client_mock)

        stale_base = planner.base_units(manager.current_step())
        manager.start_quorum()
        plan = planner.plan()
        # healed before the plan: it trains step 20 with every replica group,
        # so the base read before the quorum is stale
        self.assertEqual(manager.wait_membership(), tuple(quorum_ids([0, OWN_SLOT])))
        self.assertEqual(manager.current_step(), 20)
        self.assertEqual(stale_base, (3, 4, 5))
        self.assertEqual(planner.base_units(20), (243, 244, 245))
        # slots 2 and 3 are absent: units 6..11, the second half is ours
        self.assertEqual(plan, (249, 250, 251))
        self.assertTrue(manager.is_participating())
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())
        self.assertEqual(manager.current_step(), 21)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_async_heal_plans_before_healing(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=True)
        planner = make_planner(manager)
        self._healing_quorum(manager, client_mock)

        manager.start_quorum()
        plan = planner.plan()
        # outside the max-step view while healing: its base only, zeroed
        self.assertEqual(manager.wait_membership(), tuple(quorum_ids([0])))
        self.assertEqual(plan, ())
        self.assertEqual(planner.base_units(0), (3, 4, 5))
        manager.wait_quorum()
        self.assertEqual(manager.current_step(), 20)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_reconfigure_error_still_plans(self, client_mock: MagicMock) -> None:
        # _async_quorum only reports a reconfigure error; plan() must still
        # return, and the step does not commit
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = RuntimeError("configure failed")
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)

        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual(
            manager.wait_membership(), tuple(quorum_ids([0, OWN_SLOT, 2]))
        )
        self.assertEqual(plan, (10,))
        self.assertIsNotNone(manager.errored())
        self.assertFalse(manager.should_commit())

    def _duplicate_slot_quorum(self) -> QuorumResult:
        quorum = self._quorum(1, 0, 2)
        quorum.replica_ids = ["replica_1:a", "replica_1:b"]
        quorum.max_replica_ids = ["replica_1:a", "replica_1:b"]
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
        self.assertEqual(manager.num_participants(), 2)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_accessors(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(min_replica_size=3)
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

        # slots 0, 1, 2: slot 3's units 9..11, one each, ours is 10
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual(manager.wait_membership(), tuple(quorum_ids([0, 1, 2])))
        self.assertEqual(plan, (10,))
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())

        # next quorum: slots 0, 1 at step 1, ours is the second half of 6..11
        client_mock()._quorum.return_value = self._quorum(2, 1, 2, step=1)
        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual(manager.wait_membership(), tuple(quorum_ids([0, OWN_SLOT])))
        self.assertEqual(manager.current_step(), 1)
        self.assertEqual(plan, (21, 22, 23))

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_fixed_with_spares(self, client_mock: MagicMock) -> None:
        # P is clamped to min_replica_size and ranks >= min become spares; a
        # spare's slot is read by the active groups
        manager = self._create_manager(
            min_replica_size=2, world_size_mode=WorldSizeMode.FIXED_WITH_SPARES
        )
        planner = make_planner(manager)

        client_mock()._quorum.return_value = self._quorum(1, 2, 3)
        manager.start_quorum()
        spare = planner.plan()
        # the clamp drops this group from the participants: the reported list
        # is the active groups' slots
        self.assertEqual(manager.wait_membership(), tuple(quorum_ids([0, 2])))
        self.assertEqual(spare, ())

        # a lower-ranked replica leaves: the same participants after the clamp,
        # and this group moves from outside the list to its second entry
        client_mock()._quorum.return_value = self._quorum(2, 1, 2)
        manager.start_quorum()
        promoted = planner.plan()
        self.assertEqual(
            manager.wait_membership(), tuple(quorum_ids([0, OWN_SLOT]))
        )
        self.assertEqual(promoted, (9, 10, 11))

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_does_not_wait_for_reconfigure(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        release = threading.Event()
        self.addCleanup(release.set)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = lambda *_: release.wait(10)

        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        planner.plan()
        self.assertEqual(
            manager.wait_membership(), tuple(quorum_ids([0, OWN_SLOT, 2]))
        )
        # the membership is out while the round is still reconfiguring
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
        self.assertEqual(manager.num_participants(), 2)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_error_reaches_plan(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        client_mock()._quorum.side_effect = RuntimeError("quorum rpc failed")
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "quorum rpc failed"):
            planner.plan()
        with self.assertRaisesRegex(RuntimeError, "quorum rpc failed"):
            manager.wait_quorum()

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_membership_before_start_quorum(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            manager.wait_membership()
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        self.assertEqual(manager.num_participants(), 3)
        assert manager._quorum_future is not None
        self.assertTrue(manager._quorum_future.done())
