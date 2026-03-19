from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.measure import find_contours

from .io import load_any_tiff, save_uint8_tiff, safe_name, valid_pixel_size, write_json
from .models import RegionParams
from .visualization import add_colored_type_text, add_scalebar_20um, axis_off


def um_to_px_iso(value_um: float, pixel_size_um: Tuple[float, float]) -> int:
    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    scale = np.sqrt(max(1e-12, px_um_x * px_um_y))
    return max(1, int(round(float(value_um) / scale)))


def build_region_mask_for_type(
    type_name: str,
    celltype_mask: np.ndarray,
    df_cells: pd.DataFrame,
    name_to_id: Dict[str, int],
    close_px: int,
    dilate_px: int,
    min_area_px: int,
    min_cells: int,
) -> np.ndarray:
    height, width = celltype_mask.shape
    type_id = name_to_id[type_name]
    region = celltype_mask == type_id

    if close_px > 0:
        if hasattr(morphology, "isotropic_closing"):
            region = morphology.isotropic_closing(region, radius=close_px)
        else:
            region = morphology.binary_closing(region, footprint=morphology.disk(close_px))

    region = ndi.binary_fill_holes(region)

    if min_area_px > 0:
        region = morphology.remove_small_objects(region, min_size=min_area_px)

    if dilate_px > 0:
        if hasattr(morphology, "isotropic_dilation"):
            region = morphology.isotropic_dilation(region, radius=dilate_px)
        else:
            region = morphology.binary_dilation(region, footprint=morphology.disk(dilate_px))

    lbl = measure.label(region, connectivity=2)
    if lbl.max() == 0:
        return np.zeros_like(region, dtype=bool)

    sub = df_cells[df_cells["celltype"].astype(str) == str(type_name)][["centroid_x_px", "centroid_y_px"]]
    if len(sub) == 0:
        return np.zeros_like(region, dtype=bool)

    cy = np.clip(np.rint(sub["centroid_y_px"].to_numpy(float)).astype(int), 0, height - 1)
    cx = np.clip(np.rint(sub["centroid_x_px"].to_numpy(float)).astype(int), 0, width - 1)
    region_ids = lbl[cy, cx]
    region_ids = region_ids[region_ids > 0]
    if region_ids.size == 0:
        return np.zeros_like(region, dtype=bool)

    counts = np.bincount(region_ids, minlength=int(lbl.max()) + 1)
    keep_ids = np.where(counts >= int(min_cells))[0]
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        return np.zeros_like(region, dtype=bool)

    return np.isin(lbl, keep_ids)


def run_region_boundary_analysis(
    df_cells: pd.DataFrame,
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    params: RegionParams,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    if not valid_pixel_size(pixel_size_um):
        raise RuntimeError("PIXEL_SIZE_UM missing/invalid; region analysis needs valid pixel size.")

    if celltype_mask.ndim != 2:
        raise RuntimeError(
            f"celltype_mask must be a 2D label image, but got shape={celltype_mask.shape}."
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    height, width = celltype_mask.shape
    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    px_area_um2 = px_um_x * px_um_y

    required_cols = {"label", "celltype", "centroid_x_px", "centroid_y_px"}
    if not required_cols.issubset(set(df_cells.columns)):
        raise RuntimeError(f"df_cells must contain columns: {required_cols}")

    ct_names = [ct["name"] for ct in celltype_cfg]
    ct_hex = [ct["color_hex"] for ct in celltype_cfg]
    cmap_ct = ListedColormap([(0, 0, 0)] + [mcolors.to_rgb(color) for color in ct_hex])
    name_to_id = {ct_names[i]: (i + 1) for i in range(len(ct_names))}

    close_px = um_to_px_iso(params.close_um, pixel_size_um)
    dilate_px = um_to_px_iso(params.dilate_um, pixel_size_um)
    min_area_px = int(round(float(params.min_area_um2) / max(1e-12, px_area_um2)))
    min_cells = int(params.min_cells)

    df_base = df_cells[["label", "celltype", "centroid_x_px", "centroid_y_px"]].copy()
    cy_all = np.clip(np.rint(df_base["centroid_y_px"].to_numpy(float)).astype(int), 0, height - 1)
    cx_all = np.clip(np.rint(df_base["centroid_x_px"].to_numpy(float)).astype(int), 0, width - 1)

    masks: Dict[str, np.ndarray] = {}
    for type_name in params.selected_types:
        mask = build_region_mask_for_type(
            type_name=type_name,
            celltype_mask=celltype_mask,
            df_cells=df_cells,
            name_to_id=name_to_id,
            close_px=close_px,
            dilate_px=dilate_px,
            min_area_px=min_area_px,
            min_cells=min_cells,
        )
        masks[type_name] = mask
        if save_outputs:
            save_uint8_tiff(save_dir / f"{safe_name(type_name, 'region')}_region_mask_uint8.tiff", mask.astype(np.uint8))

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 8))
    ax.imshow(celltype_mask, cmap=cmap_ct, origin="upper", interpolation="nearest", vmin=0, vmax=len(ct_names))
    axis_off(ax)

    ds = int(params.contour_downsample)
    for type_name in params.selected_types:
        mask = masks[type_name]
        if not np.any(mask):
            continue
        if ds > 1:
            mask_ds = mask[::ds, ::ds]
            contours = find_contours(mask_ds.astype(float), 0.5)
        else:
            contours = find_contours(mask.astype(float), 0.5)

        boundary_color = (
            celltype_cfg[name_to_id[type_name] - 1]["color_hex"] if params.use_type_colors else params.boundary_color
        )
        for contour in contours:
            x = contour[:, 1] * ds
            y = contour[:, 0] * ds
            ax.plot(x, y, params.line_style, color=boundary_color, linewidth=float(params.line_width), label="_nolegend_")

    title = "Boundaries: " + ", ".join(params.selected_types)
    add_colored_type_text(ax, ct_names, ct_hex, title=title, fontsize=14)
    add_scalebar_20um(ax, celltype_mask.shape, px_um_x, bar_um=20.0, color="white", lw=4, pad_frac=0.05)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    base_all = safe_name("__".join(params.selected_types), "region")
    out_svg = save_dir / f"celltypes_with_boundaries__{base_all}.svg"
    out_tiff = save_dir / f"celltypes_with_boundaries__{base_all}.tiff"
    if save_outputs:
        fig.savefig(out_svg, dpi=600, bbox_inches="tight", pad_inches=0)
        fig.savefig(out_tiff, dpi=600, bbox_inches="tight", pad_inches=0)

    long_rows: List[pd.DataFrame] = []
    for type_name in params.selected_types:
        mask = masks[type_name]
        inside = mask[cy_all, cx_all] if np.any(mask) else np.zeros_like(cy_all, dtype=bool)
        tmp = df_base.copy()
        tmp["boundary_type"] = type_name
        tmp["region"] = np.where(inside, f"{type_name}_region", "adjacent_region")
        long_rows.append(tmp)

    df_assign = pd.concat(long_rows, ignore_index=True)
    counts_by_region = (
        df_assign.groupby(["boundary_type", "region", "celltype"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["boundary_type", "region", "celltype"])
        .reset_index(drop=True)
    )

    counts_csv = save_dir / f"celltype_counts_by_region__{base_all}.csv"
    assign_csv = save_dir / f"cell_region_assignments__{base_all}.csv"
    params_json = save_dir / f"region_params__{base_all}.json"
    if save_outputs:
        counts_by_region.to_csv(counts_csv, index=False)
        df_assign.to_csv(assign_csv, index=False)
        write_json(params_json, params.to_dict())

    return {
        "masks": masks,
        "figure": fig,
        "counts_by_region": counts_by_region,
        "assignments": df_assign,
        "saved_paths": {
            "overlay_svg": out_svg,
            "overlay_tiff": out_tiff,
            "counts_csv": counts_csv,
            "assignments_csv": assign_csv,
            "params_json": params_json,
        },
    }


def discover_boundary_mask_files(save_dir: Path, celltype_cfg: Sequence[Dict[str, Any]], df_cells: pd.DataFrame) -> List[Tuple[str, Path]]:
    save_dir = Path(save_dir)
    ct_names = [ct["name"] for ct in celltype_cfg]
    present_types = sorted(set(df_cells["celltype"].astype(str)))
    ct_names = [ct for ct in ct_names if ct in present_types] or present_types

    mask_candidates: List[Tuple[str, Path]] = []
    for name in ct_names:
        path = save_dir / f"{safe_name(name, 'region')}_region_mask_uint8.tiff"
        if path.exists():
            mask_candidates.append((name, path))

    extra = sorted(save_dir.glob("*_region_mask_uint8.tiff"))
    known_paths = {path for _, path in mask_candidates}
    for path in extra:
        if path not in known_paths:
            mask_candidates.append((path.stem.replace("_region_mask_uint8", ""), path))

    return mask_candidates
