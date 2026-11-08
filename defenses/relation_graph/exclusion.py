"""Confirmed-group zero-weight variant of the relation-graph defense.

The base defense always keeps at least ``1 - attenuation`` of every client's
nominal weight.  This variant removes every principal that belongs to a
confirmed anomalous group entirely; the existing anchor (the model broadcast
at the start of the round) receives the removed mass.  Detection, graph
evidence, history and checkpointing are inherited unchanged, so the only
difference is the action rule.

Validated against ``RelationGraphDefense.state_dict``: no extra state is
introduced, and checkpoints remain interchangeable with the base defense.
"""
from __future__ import annotations

import numpy as np

from defenses.relation_graph.defense import RelationGraphDefense


class GroupExclusionRelationGraphDefense(RelationGraphDefense):
    """Zero the aggregation weight of every confirmed-group principal."""

    def _removal(self, nominal, risks, assessment):
        removed = super()._removal(nominal, risks, assessment)
        members = {
            principal
            for group in assessment.groups
            for principal in group["principals"]
        }
        if members:
            mask = np.asarray([principal in members for principal in self._principals])
            removed = np.where(mask, nominal, removed)
        return removed