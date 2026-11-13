# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Elastic Batch
=============

This module splits a fixed global batch across a changing set of replica
groups.

The job is deployed with ``N`` replica groups. Each one owns a slot in
``[0, N)``, the trailing digits of its ``replica_id``, and every step the
slot owns ``m`` consecutive rank-batches of the global sample stream (one
rank-batch is one micro-batch on every data-parallel rank of a group). So a
step holds ``U = N * m`` rank-batches and the global batch is
``U * local_data_parallel_size * micro_batch_size`` samples.

A replica group's own ``m`` rank-batches (its *base*) depend only on the step
and its slot, so it can read and train on them before the quorum returns:
:py:meth:`ElasticBatchPlanner.base_units` gives them without waiting for
anything. The rank-batches of the slots missing from the quorum (the *tail*)
are split among the participants once membership is known: the planner
registers :py:meth:`ElasticBatchPlanner.plan_from_view` with
:py:meth:`Manager.set_plan_fn`, which computes it on the quorum thread before
ProcessGroup reconfiguration and healing.

Every group, participating or not, runs at least its ``m`` base micro-batches;
a non-participant's gradient is zeroed by the Manager, and its slot is part of
the tail. Scale each micro-batch loss by ``1 / U`` and SUM the gradients
across replica groups: every sample then weighs ``1 / global_batch_size``
whatever the membership.

Requirements:

* ``init_sync=False`` on the Manager.
* Every replica group's ``replica_id`` ends in its slot, and no two live
  groups share a slot: a replacement group reuses the slot it replaces.
* With ``use_async_quorum=True`` (the default) a healing replica group is
  outside the participants, so the others read its slot. A synchronous quorum
  that allows healing heals it before the forward pass and plans after
  healing, so it participates with the healed step; its base must then be
  read for that step (:py:meth:`ElasticBatchPlanner.base_units`).
* The dataset supports random access by index.
"""

import re
from typing import cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchft.manager import Manager, QuorumView


# (participant slots in participating_rank order, idx): the assignment key.
# quorum_id and the per-process UUIDs do not affect the assignment, so a
# quorum bump that leaves both unchanged does not replan.
_PlanKey = Tuple[Tuple[int, ...], Optional[int]]

_TRAILING_DIGITS = re.compile(r"(\d+)$")


def replica_slot(replica_id: str) -> int:
    """
    Return the slot of a replica group: the trailing digits of its
    ``replica_id``, ignoring the ``:<uuid>`` suffix the quorum adds.

    Raises:
        ValueError: if the replica_id does not end in digits
    """
    match = _TRAILING_DIGITS.search(replica_id.split(":")[0])
    if match is None:
        raise ValueError(f"replica_id {replica_id!r} does not end in a slot number")
    return int(match.group(1))


class ElasticBatchPlanner:
    """
    Assigns this replica group its rank-batches for each step.

    The constructor registers the planner with
    :py:meth:`Manager.set_plan_fn`, so the tail is computed on the quorum
    thread as soon as membership is known. A Manager holds one planner;
    construct it before the first ``manager.start_quorum()``.

    Each step: read :py:meth:`base_units` for ``manager.current_step()``
    (no wait), start the quorum, run all but the last base micro-batch, then
    :py:meth:`plan` (waits for membership only) gives the tail units, whose
    count tells which micro-batch is the last.

    Both accessors return unit numbers: the rank-batches of the global sample
    stream at :py:func:`local_sample_indices`. A step's work is
    ``base_units(step) + plan()``, and over all participants the two cover the
    ``U`` units of that step once, with none left over.

    Args:
        manager: the torchft Manager; its ``replica_id()`` gives the slot.
        num_replicas: the number of slots (N), the most replica groups that
            participate at once.
        base_grad_accum: micro-batches per rank that each slot owns per step
            (m).
    """

    def __init__(
        self,
        manager: "Manager",
        num_replicas: int,
        base_grad_accum: int,
    ) -> None:
        self._manager = manager
        self._N = num_replicas
        self._m = base_grad_accum
        min_replica_size = manager.min_replica_size()

        context: Dict[str, int] = {
            "num_replicas": num_replicas,
            "base_grad_accum": base_grad_accum,
            "min_replica_size": min_replica_size,
        }
        for name, value in context.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1 ({context})")
        if self._N - 1 < min_replica_size:
            raise ValueError(
                "num_replicas - 1 must be >= min_replica_size so that one "
                f"failure does not block commits ({context})"
            )

        replica_id = manager.replica_id()
        if not replica_id:
            raise ValueError("the Manager needs a replica_id ending in its slot")
        self._slot: int = replica_slot(replica_id)
        if not 0 <= self._slot < self._N:
            raise ValueError(
                f"slot {self._slot} of replica_id {replica_id!r} is outside "
                f"[0, num_replicas) ({context})"
            )

        self._U: int = self._N * self._m

        self._cached_key: Optional[_PlanKey] = None
        # the tail as offsets within a step for _cached_key; written only on
        # the quorum thread, by plan_from_view
        self._cached_tail: Tuple[int, ...] = ()

        manager.set_plan_fn(self.plan_from_view)

    @property
    def U(self) -> int:
        """Rank-batches per step across all replica groups: ``N * m``."""
        return self._U

    def base_units(self, step: int) -> Tuple[int, ...]:
        """
        This group's own units at ``step``. Needs no quorum, so the data can be
        fetched before ``start_quorum()``.
        """
        start = step * self._U + self._slot * self._m
        return tuple(range(start, start + self._m))

    def plan_from_view(self, view: "QuorumView") -> Tuple[int, ...]:
        """
        Compute the tail units for one quorum. The constructor registers this
        with :py:meth:`Manager.set_plan_fn`, so it runs on the quorum thread at
        the membership point; it only does arithmetic.

        Recomputes the tail only when the participant slots or idx changed.
        """
        P = view.num_participants
        if P == 0:
            raise RuntimeError(f"no quorum (quorum_id={view.quorum_id})")
        ids = view.participating_replica_ids
        if len(ids) != P:
            raise RuntimeError(
                f"{len(ids)} participating replica ids for {P} participants; "
                f"the Manager must report max_replica_ids (quorum_id={view.quorum_id})"
            )
        key = (tuple(replica_slot(i) for i in ids), view.participating_rank)
        if key != self._cached_key:
            self._assign(key, view.quorum_id)
        return self._build(view)

    def _assign(self, key: _PlanKey, quorum_id: int) -> None:
        slots, idx = key
        present = set(slots)
        if len(present) != len(slots) or not present <= set(range(self._N)):
            raise RuntimeError(
                f"participant slots {list(slots)} must be distinct and in "
                f"[0, {self._N}) (quorum_id={quorum_id})"
            )
        if idx is not None and slots[idx] != self._slot:
            raise RuntimeError(
                f"participating_rank {idx} is slot {slots[idx]}, not this "
                f"group's slot {self._slot} (quorum_id={quorum_id})"
            )
        if idx is None:
            self._cached_tail = ()
        else:
            absent = [
                slot * self._m + j
                for slot in range(self._N)
                if slot not in present
                for j in range(self._m)
            ]
            q, r = divmod(len(absent), len(slots))
            start = idx * q + min(idx, r)
            self._cached_tail = tuple(
                absent[start : start + q + (1 if idx < r else 0)]
            )
        self._cached_key = key

    def _build(self, view: "QuorumView") -> Tuple[int, ...]:
        # the step the quorum was asked at (or healed to): it does not advance
        # on a failed commit, so a retried step reads the same units
        offset = view.step * self._U
        return tuple(offset + u for u in self._cached_tail)

    def plan(self) -> Tuple[int, ...]:
        """
        The tail units of the current step.

        This waits for the in-flight quorum's membership, not for ProcessGroup
        reconfiguration or healing. Empty when every slot is present, or when
        this group does not participate.
        """
        return cast(Tuple[int, ...], self._manager.wait_plan())


def local_sample_indices(
    units: Sequence[int],
    micro_batch_size: int,
    local_dp_rank: int,
    local_dp_size: int,
) -> List[int]:
    """
    The global sample stream indices one data-parallel rank reads for
    ``units``, one micro-batch per unit in order.

    Data-parallel rank ``d`` reads samples ``[u * rb + d * micro, u * rb +
    (d + 1) * micro)`` of unit ``u``. Ranks that read identical data
    (tensor/context parallel) must pass the same ``local_dp_rank``.
    """
    assert 0 <= local_dp_rank < local_dp_size, (local_dp_rank, local_dp_size)
    rank_batch = local_dp_size * micro_batch_size
    indices: List[int] = []
    for unit in units:
        start = unit * rank_batch + local_dp_rank * micro_batch_size
        indices.extend(range(start, start + micro_batch_size))
    return indices


class DatasetIndexMapper:
    """
    Maps the global sample stream onto dataset indices.

    The stream is one random permutation of the dataset per epoch, laid end to
    end, so a window may span an epoch boundary. The permutation depends only
    on ``seed`` and the epoch, so every replica group gets the same mapping
    without communicating.

    Args:
        dataset_len: the number of samples in the dataset; must not change
            during training.
        seed: the base shuffle seed.
        shuffle: if False the stream is the dataset in order, repeated.
    """

    def __init__(self, dataset_len: int, seed: int = 0, shuffle: bool = True) -> None:
        if dataset_len < 1:
            raise ValueError("dataset_len must be >= 1")
        self._len = dataset_len
        self._seed = seed
        self._shuffle = shuffle
        self._epoch: Optional[int] = None
        self._perm: Optional[torch.Tensor] = None

    def _perm_for(self, epoch: int) -> torch.Tensor:
        if self._epoch != epoch or self._perm is None:
            generator = torch.Generator().manual_seed(self._seed + epoch)
            self._perm = torch.randperm(self._len, generator=generator)
            self._epoch = epoch
        return self._perm

    def __call__(self, global_index: int) -> int:
        if global_index < 0:
            raise ValueError("global_index must be >= 0")
        epoch, offset = divmod(global_index, self._len)
        if not self._shuffle:
            return offset
        return int(self._perm_for(epoch)[offset])
