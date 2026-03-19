from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ChannelConfig:
    file: str
    channel: str
    color_hex: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfig:
    folder: Path
    save_dir: Path
    pixel_size_um: Tuple[float, float]
    image_id: str = "FieldA"
    channels: List[ChannelConfig] = field(default_factory=list)
    overlay_channels: List[str] = field(default_factory=list)
    white_channel: Optional[str] = None
    white_weight: float = 0.0
    input_mode: str = "local"

    def to_notebook_cfg(self) -> Dict[str, Any]:
        return {
            "FOLDER": self.folder,
            "SAVE_DIR": self.save_dir,
            "PIXEL_SIZE_UM": self.pixel_size_um,
            "IMAGE_ID": self.image_id,
            "CHANNELS": [c.to_dict() for c in self.channels],
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "folder": str(self.folder),
            "save_dir": str(self.save_dir),
            "pixel_size_um": [float(self.pixel_size_um[0]), float(self.pixel_size_um[1])],
            "image_id": self.image_id,
            "channels": [c.to_dict() for c in self.channels],
            "overlay_channels": list(self.overlay_channels),
            "white_channel": self.white_channel,
            "white_weight": float(self.white_weight),
            "input_mode": self.input_mode,
        }


@dataclass
class NucleiParams:
    nucleus_channel: str
    min_diam_um: float = 6.0
    max_diam_um: float = 60.0
    tophat_radius_um: float = 2.0
    gauss_sigma_um: float = 0.5
    local_win_um: float = 25.0
    local_offset: float = -0.03
    h_maxima_um: float = 0.25
    seed_min_dist_um: float = 0.1
    watershed_compactness: float = 0.5
    post_resplit_mult: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "NUCLEUS_CHANNEL": self.nucleus_channel,
            "MIN_DIAM_UM": float(self.min_diam_um),
            "MAX_DIAM_UM": float(self.max_diam_um),
            "TOPHAT_RADIUS_UM": float(self.tophat_radius_um),
            "GAUSS_SIGMA_UM": float(self.gauss_sigma_um),
            "LOCAL_WIN_UM": float(self.local_win_um),
            "LOCAL_OFFSET": float(self.local_offset),
            "H_MAXIMA_UM": float(self.h_maxima_um),
            "SEED_MIN_DIST_UM": float(self.seed_min_dist_um),
            "WATERSHED_COMPACTNESS": float(self.watershed_compactness),
            "POST_RESPLIT_MULT": float(self.post_resplit_mult),
        }


@dataclass
class RegionParams:
    selected_types: List[str]
    close_um: float = 15.0
    dilate_um: float = 10.0
    min_area_um2: float = 20000.0
    min_cells: int = 5
    contour_downsample: int = 2
    line_width: float = 2.0
    line_style: str = "--"
    boundary_color: str = "#a1d99b"
    use_type_colors: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_types": list(self.selected_types),
            "close_um": float(self.close_um),
            "dilate_um": float(self.dilate_um),
            "min_area_um2": float(self.min_area_um2),
            "min_cells": int(self.min_cells),
            "contour_downsample": int(self.contour_downsample),
            "line_width": float(self.line_width),
            "line_style": str(self.line_style),
            "boundary_color": str(self.boundary_color),
            "use_type_colors": bool(self.use_type_colors),
        }


CellTypeConfig = Dict[str, Any]


def make_channel_configs(rows: Sequence[Dict[str, Any]]) -> List[ChannelConfig]:
    out: List[ChannelConfig] = []
    for row in rows:
        out.append(
            ChannelConfig(
                file=str(row["file"]),
                channel=str(row["channel"]),
                color_hex=str(row["color_hex"]),
            )
        )
    return out
