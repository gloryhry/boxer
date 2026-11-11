"""Read and validate the potential-field defaults for each new job."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT = Path(__file__).resolve().parents[2]


MAX_SAFE_INTEGER = 2**53 - 1


class PotentialFieldConfigError(ValueError):
    """The configured YAML could not be read or validated."""


class PotentialFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    voxel: float = Field(ge=.02, le=5, allow_inf_nan=False)
    vis_targets: int = Field(ge=16, le=2048)
    min_room_points: int = Field(ge=0, le=100_000_000)
    min_room_area: float = Field(ge=0, le=1_000_000, allow_inf_nan=False)
    merge_opening: float = Field(ge=0, le=10, allow_inf_nan=False)
    max_projection_distance: float = Field(ge=0, le=10, allow_inf_nan=False)
    ceiling_aware: bool
    max_grid_voxels: int = Field(ge=1, le=MAX_SAFE_INTEGER)
    max_stacks: int = Field(ge=1, le=MAX_SAFE_INTEGER)


class _ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        values = {}
        for key, value in self.construct_pairs(node, deep=deep):
            if not isinstance(key, str):
                raise ValueError("配置键必须为字符串")
            if key in values:
                raise ValueError(f"重复的配置键 {key}")
            values[key] = value
        return values


def load_potential_field_config():
    """Read a fresh snapshot. Relative overrides resolve from the checkout."""
    path = Path(os.environ.get("BOXER_PF_CONFIG")
                or os.environ.get("ROOMSEG_PF_CONFIG")
                or "config/potential_field.yaml")
    try:
        path = PROJECT / path.expanduser()
        values = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
        return PotentialFieldConfig.model_validate(values).model_dump()
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        raise PotentialFieldConfigError(f"势场配置无效（{path}）\n{error}") from error
