from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import ChannelConfig, PipelineConfig


def set_default_thread_env() -> Dict[str, str]:
    """
    Mirror the notebook's Cell 1 environment setup as closely as possible.
    This should be called before importing heavy numerical libraries in entrypoints
    such as app.py. The function is kept here for notebook/demo reuse as well.
    """
    import os

    cpu = os.cpu_count() or 4
    n_threads = max(1, cpu - 1)

    defaults = {
        "NUMBA_THREADING_LAYER": "workqueue",
        "OMP_NUM_THREADS": str(n_threads),
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "MKL_NUM_THREADS": str(n_threads),
        "OPENBLAS_NUM_THREADS": str(n_threads),
        "NUMEXPR_NUM_THREADS": str(n_threads),
        "KMP_WARNINGS": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    return {k: os.environ.get(k, "") for k in defaults}


def resolve_folder(folder_value: str, root: Path | None = None) -> Path:
    value = (folder_value or "").strip()
    if not value:
        return (root or Path.cwd()).resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = (root or Path.cwd()).resolve()
    return (base / path).resolve()


def discover_text_image_files(folder: Path) -> List[str]:
    exts = {".csv", ".txt"}
    return sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def stem_no_ext(filename: str) -> str:
    return Path(filename).stem


def valid_pixel_size(pixel_size_um: Tuple[float, float] | None) -> bool:
    try:
        return (
            pixel_size_um is not None
            and len(pixel_size_um) == 2
            and float(pixel_size_um[0]) > 0
            and float(pixel_size_um[1]) > 0
        )
    except Exception:
        return False


def load_text_grid(path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    with open(path, "r", errors="ignore") as handle:
        first = handle.readline()
    if "\t" in first:
        sep = "\t"
    elif "," in first:
        sep = ","
    elif ";" in first:
        sep = ";"
    else:
        sep = r"\s+"
    df = pd.read_csv(path, header=None, sep=sep, engine="python" if sep == r"\s+" else "c")
    return df.to_numpy(dtype=dtype, copy=False)


def files_to_long_df(
    folder: Path,
    channels_cfg: Sequence[Dict[str, Any]] | Sequence[ChannelConfig],
    image_id: str = "FieldA",
    pixel_size_um: Tuple[float, float] | None = None,
    unit: str = "um",
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Tuple[int, int]]]:
    long_frames: List[pd.DataFrame] = []
    shapes: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for channel_cfg in channels_cfg:
        if isinstance(channel_cfg, ChannelConfig):
            file_name = channel_cfg.file
            channel_name = channel_cfg.channel
        else:
            file_name = str(channel_cfg["file"])
            channel_name = str(channel_cfg["channel"])

        path = folder / file_name
        arr = load_text_grid(path)
        h, w = arr.shape
        y_idx, x_idx = np.indices((h, w))

        df = pd.DataFrame(
            {
                "image_id": image_id,
                "channel": channel_name,
                "y_px": y_idx.ravel(order="C").astype(np.int32),
                "x_px": x_idx.ravel(order="C").astype(np.int32),
                "value": arr.ravel(order="C"),
            }
        )
        if pixel_size_um is not None:
            df[f"x_{unit}"] = df["x_px"] * float(pixel_size_um[0])
            df[f"y_{unit}"] = df["y_px"] * float(pixel_size_um[1])

        long_frames.append(df)
        shapes[(image_id, channel_name)] = (h, w)

    return pd.concat(long_frames, ignore_index=True), shapes


def channels_by_image(df: pd.DataFrame) -> pd.Series:
    return df.groupby("image_id")["channel"].unique().apply(lambda x: list(map(str, x)))


def to_image(df: pd.DataFrame, shapes: Dict[Tuple[str, str], Tuple[int, int]], image_id: str, channel: str) -> np.ndarray:
    key = (image_id, channel)
    if key not in shapes:
        available = channels_by_image(df).get(image_id, [])
        raise KeyError(f"Missing {key}. Available channels: {available}")
    h, w = shapes[key]
    sub = df[(df["image_id"] == image_id) & (df["channel"] == channel)]
    return sub["value"].to_numpy().reshape(h, w)


def load_any_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return tifffile.imread(str(path))
    except Exception:
        import imageio.v2 as imageio

        return imageio.imread(str(path))


def save_uint16_tiff(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_u16 = arr.astype(np.uint16)
    try:
        import tifffile

        tifffile.imwrite(str(path), arr_u16)
        return
    except Exception:
        pass

    try:
        from PIL import Image

        Image.fromarray(arr_u16).save(str(path))
        return
    except Exception:
        pass

    import imageio.v2 as imageio

    imageio.imwrite(str(path), arr_u16)


def save_uint8_tiff(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_u8 = arr.astype(np.uint8)
    try:
        import tifffile

        tifffile.imwrite(str(path), arr_u8)
        return
    except Exception:
        pass

    import imageio.v2 as imageio

    imageio.imwrite(str(path), arr_u8)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def pipeline_config_to_json_dict(config: PipelineConfig) -> Dict[str, Any]:
    return config.to_json_dict()


def safe_name(text: str, fallback: str = "item") -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_") or fallback


def list_output_files(folder: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not folder.exists():
        return rows
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "name": path.name,
                    "relative_path": str(path.relative_to(folder)),
                    "size_bytes": int(stat.st_size),
                }
            )
    return rows


def zip_directory_bytes(folder: Path) -> bytes:
    if not folder.exists():
        return b""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(folder)))
    buffer.seek(0)
    return buffer.getvalue()


def save_uploaded_file_bytes(file_name: str, file_bytes: bytes, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_file_name = Path(file_name).name
    out_path = destination_dir / safe_file_name
    out_path.write_bytes(file_bytes)
    return out_path
