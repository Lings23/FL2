"""Frozen model-metadata resolution layout for RTC-v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


HEAD_ROLES = {"head", "classifier_head", "classification_head"}


@dataclass(frozen=True)
class ResolutionLayout:
    indices: Mapping[str, tuple[int, ...]]

    @classmethod
    def build(
        cls,
        params: Sequence[np.ndarray],
        roles: Mapping[int | str, str],
        enabled: Sequence[str],
        *,
        max_blocks: int = 4,
    ) -> "ResolutionLayout":
        role_map = {str(key): str(value).lower() for key, value in roles.items()}
        floating = tuple(
            index
            for index, param in enumerate(params)
            if np.issubdtype(np.asarray(param).dtype, np.floating)
        )
        if not floating:
            raise ValueError("RTC-v3 resolution layout requires floating parameters")
        enabled_names = tuple(str(value) for value in enabled)
        if not enabled_names or enabled_names[0] != "full":
            raise ValueError("RTC-v3 resolutions must start with 'full'")

        head = tuple(
            index
            for index in floating
            if role_map.get(str(index), "unassigned") in HEAD_ROLES
        )
        non_head = [index for index in floating if index not in set(head)]
        ranked = sorted(
            non_head,
            key=lambda index: (-int(np.asarray(params[index]).size), index),
        )
        requested_blocks = sorted(
            int(name.split(":", 1)[1])
            for name in enabled_names
            if name.startswith("block:")
        )
        if requested_blocks and (
            requested_blocks != list(range(max(requested_blocks) + 1))
            or max(requested_blocks) >= max_blocks
        ):
            raise ValueError(
                f"RTC-v3 block resolutions must be contiguous block:0.."
                f"block:{max_blocks - 1}"
            )
        canonical_order = ["full"]
        if "head" in enabled_names:
            canonical_order.append("head")
        canonical_order.extend(
            f"block:{block_number}" for block_number in requested_blocks
        )
        if "overflow" in enabled_names:
            canonical_order.append("overflow")
        if tuple(canonical_order) != enabled_names:
            raise ValueError(
                "RTC-v3 resolutions must be ordered as "
                "full, head, contiguous blocks, overflow"
            )

        layout: dict[str, tuple[int, ...]] = {"full": floating}
        selected_blocks: set[int] = set()
        for name in enabled_names[1:]:
            if name == "head":
                if not head:
                    raise ValueError(
                        "RTC-v3 head resolution requires classifier-head metadata"
                    )
                layout[name] = head
            elif name.startswith("block:"):
                block_number = int(name.split(":", 1)[1])
                if block_number >= len(ranked):
                    raise ValueError(f"RTC-v3 {name} has no corresponding tensor")
                index = ranked[block_number]
                layout[name] = (index,)
                selected_blocks.add(index)
            elif name == "overflow":
                overflow = tuple(
                    index
                    for index in non_head
                    if index not in selected_blocks
                )
                if not overflow:
                    raise ValueError(
                        "RTC-v3 overflow resolution has no remaining parameters"
                    )
                layout[name] = overflow
            else:
                raise ValueError(f"unknown RTC-v3 resolution {name!r}")
        return cls(indices=layout)

    def tensor_positions(
        self,
        resolution: str,
        floating_indices: Sequence[int],
    ) -> tuple[int, ...]:
        positions = {parameter_index: position for position, parameter_index in enumerate(
            floating_indices
        )}
        return tuple(positions[index] for index in self.indices[resolution])


def squared_norm(arrays: Sequence[np.ndarray], positions: Sequence[int]) -> float:
    return float(
        sum(
            np.dot(
                np.asarray(arrays[position], dtype=np.float64).reshape(-1),
                np.asarray(arrays[position], dtype=np.float64).reshape(-1),
            )
            for position in positions
        )
    )
