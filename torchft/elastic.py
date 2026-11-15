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

The job is deployed with ``N`` replica groups, each owning a slot in
``[0, N)`` (the trailing digits of its ``replica_id``). Every step a slot owns
``m`` consecutive units, so a step holds ``U = N * m`` units and the global
batch is ``U * local_data_parallel_size * micro_batch_size`` samples. A *unit*
is the ordinal of one micro-batch on every data-parallel rank's sample stream;
which samples a unit holds is the caller's choice, it only has to be the same
function of the unit number on every replica group.

A group's own ``m`` units (its *base*) depend only on the step and its slot,
so :py:meth:`ElasticBatchPlanner.base_units` reads them without a quorum. The
units of the missing slots (the *tail*) are split among the participants once
membership is known.

Every group runs its base, participating or not; the Manager zeroes the
gradient of a non-participant. Scale each micro-batch loss by ``1 / U`` and SUM
the gradients across groups, so every sample weighs ``1 / global_batch_size``
whatever the membership.

Requirements:

* ``init_sync=False`` on the Manager.
* A group's ``replica_id`` ends in its slot, and no two live groups share a
  slot: a replacement group reuses the slot it replaces.
* With ``use_async_quorum=True`` (the default) a healing group is outside the
  participants, so the others read its slot. A synchronous quorum heals it
  before the forward pass, so it participates with the healed step and its
  base must be read for that step.
* The dataset supports random access by row.
"""

import re
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from torchft.manager import Manager


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
    Assigns this replica group its units for each step:
    :py:meth:`base_units` needs no quorum, :py:meth:`plan` waits for the
    membership of the in-flight quorum only.

    A step's work is ``base_units(step) + plan()``, and over all participants
    the two cover the ``U`` units of that step once, with none left over.

    Args:
        manager: the torchft Manager, consulted for membership and the step.
        replica_id: this replica group's replica_id; its trailing digits are
            the slot.
        num_replicas: the number of slots (N), the most replica groups that
            participate at once.
        base_grad_accum: micro-batches per rank that each slot owns per step
            (m).
    """

    def __init__(
        self,
        manager: "Manager",
        replica_id: str,
        num_replicas: int,
        base_grad_accum: int,
    ) -> None:
        self._manager = manager
        self._N = num_replicas
        self._m = base_grad_accum

        context: Dict[str, int] = {
            "num_replicas": num_replicas,
            "base_grad_accum": base_grad_accum,
        }
        for name, value in context.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1 ({context})")

        if not replica_id:
            raise ValueError("replica_id must end in the group's slot")
        self._slot: int = replica_slot(replica_id)
        if not 0 <= self._slot < self._N:
            raise ValueError(
                f"slot {self._slot} of replica_id {replica_id!r} is outside "
                f"[0, num_replicas) ({context})"
            )

        self._U: int = self._N * self._m

    @property
    def U(self) -> int:
        """Units per step across all replica groups: ``N * m``."""
        return self._U

    def base_units(self, step: int) -> Tuple[int, ...]:
        """
        This group's own units at ``step``. Needs no quorum, so the data can be
        fetched before ``start_quorum()``.
        """
        start = step * self._U + self._slot * self._m
        return tuple(range(start, start + self._m))

    def plan(self) -> Tuple[int, ...]:
        """
        The tail units of the current step. Waits for the in-flight quorum's
        membership, not for ProcessGroup reconfiguration or healing; empty when
        every slot is present or this group does not participate.
        """
        ids = self._manager.wait_membership()
        # Position in the report, not the replica rank: the report lists the
        # replicas at max_step first. Every participant derives it from the
        # same ids, so all agree.
        slots = tuple(replica_slot(i) for i in ids)
        # a duplicate slot would silently give a group the wrong share
        if len(set(slots)) != len(slots) or not set(slots) <= set(range(self._N)):
            raise RuntimeError(
                f"participant slots {list(slots)} must be distinct and in "
                f"[0, {self._N})"
            )
        if self._slot not in slots:
            return ()
        idx = slots.index(self._slot)
        # the units of the absent slots as offsets within the step, split into
        # contiguous shares in report order, the first r one unit longer
        present = set(slots)
        absent = [
            slot * self._m + j
            for slot in range(self._N)
            if slot not in present
            for j in range(self._m)
        ]
        q, r = divmod(len(absent), len(slots))
        start = idx * q + min(idx, r)
        tail = absent[start : start + q + (1 if idx < r else 0)]
        # a failed commit does not advance the step, so a retry reads the same
        # units
        offset = self._manager.current_step() * self._U
        return tuple(offset + u for u in tail)
