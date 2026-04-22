
from __future__ import annotations

import json
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
from skimage import measure, morphology, segmentation
from skimage.measure import find_contours

from .io import load_any_tiff, save_uint8_tiff, save_uint16_tiff, safe_name, valid_pixel_size, write_json
from .models import RegionParams
from .visualization import add_colored_type_text, add_scalebar_20um, axis_off


def um_to_px_iso(value_um: float, pixel_size_um: Tuple[float, float]) -> int:
    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    scale = np.sqrt(max(1e-12, px_um_x * px_um_y))
    return max(1, int(round(float(value_um) / scale)))


def _name_to_id(celltype_cfg: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {str(ct["name"]): (i + 1) for i, ct in enumerate(celltype_cfg)}


def _name_to_color(celltype_cfg: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {str(ct["name"]): str(ct["color_hex"]) for ct in celltype_cfg}


def _selected_mask_view(
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    selected_types: Sequence[str] | None = None,
) -> np.ndarray:
    if not selected_types:
        return celltype_mask.astype(np.uint16)
    selected_ids = {i + 1 for i, ct in enumerate(celltype_cfg) if ct["name"] in selected_types}
    out = np.zeros_like(celltype_mask, dtype=np.uint16)
    keep = np.isin(celltype_mask, list(selected_ids))
    out[keep] = celltype_mask[keep].astype(np.uint16)
    return out


def make_celltype_mask_rgb(
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    selected_types: Sequence[str] | None = None,
    background_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    view_mask = _selected_mask_view(celltype_mask, celltype_cfg, selected_types)
    rgb = np.zeros((*view_mask.shape, 3), dtype=float)
    rgb[...] = np.array(background_rgb, dtype=float)
    for i, ct in enumerate(celltype_cfg, start=1):
        if selected_types and ct["name"] not in selected_types:
            continue
        rgb[view_mask == i] = np.array(mcolors.to_rgb(ct["color_hex"]), dtype=float)
    return np.clip(rgb, 0.0, 1.0)


def _mask_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    boundary = segmentation.find_boundaries(mask.astype(bool), mode="thick")
    if thickness > 1:
        boundary = morphology.binary_dilation(boundary, morphology.disk(max(1, int(thickness) - 1)))
    return boundary


def make_region_canvas_rgb(
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    masks: Dict[str, np.ndarray],
    selected_types: Sequence[str],
    display_celltypes: Sequence[str] | None = None,
    boundary_color: str = "#a1d99b",
    use_type_colors: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    rgb = make_celltype_mask_rgb(
        celltype_mask,
        celltype_cfg,
        selected_types=(display_celltypes if display_celltypes is not None else selected_types),
    )
    name_to_color = _name_to_color(celltype_cfg)
    for type_name in selected_types:
        mask = masks.get(type_name)
        if mask is None:
            continue
        boundary = _mask_boundary(mask, thickness=thickness)
        color = name_to_color.get(type_name, boundary_color) if use_type_colors else boundary_color
        rgb[boundary] = np.array(mcolors.to_rgb(color), dtype=float)
    return np.clip(rgb, 0.0, 1.0)


def _plot_mask_contours(
    ax: plt.Axes,
    mask: np.ndarray,
    color: str,
    line_width: float,
    line_style: str,
    contour_downsample: int = 1,
) -> None:
    ds = max(1, int(contour_downsample))
    if ds > 1:
        work = mask[::ds, ::ds]
        contours = find_contours(work.astype(float), 0.5)
    else:
        contours = find_contours(mask.astype(float), 0.5)

    for contour in contours:
        x = contour[:, 1] * ds
        y = contour[:, 0] * ds
        ax.plot(x, y, line_style, color=color, linewidth=float(line_width), label="_nolegend_")


def make_region_overlay_figure(
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    masks: Dict[str, np.ndarray],
    selected_types: Sequence[str],
    pixel_size_um: Tuple[float, float],
    display_celltypes: Sequence[str] | None = None,
    title: str | None = None,
    line_width: float = 2.0,
    line_style: str = "-",
    boundary_color: str = "#a1d99b",
    use_type_colors: bool = True,
    contour_downsample: int = 1,
) -> plt.Figure:
    ct_names = [ct["name"] for ct in celltype_cfg]
    ct_hex = [ct["color_hex"] for ct in celltype_cfg]
    cmap_ct = ListedColormap([(0, 0, 0)] + [mcolors.to_rgb(color) for color in ct_hex])

    display_mask = _selected_mask_view(celltype_mask, celltype_cfg, display_celltypes if display_celltypes is not None else selected_types)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 8))
    ax.imshow(display_mask, cmap=cmap_ct, origin="upper", interpolation="nearest", vmin=0, vmax=len(ct_names))
    axis_off(ax)

    name_to_color = _name_to_color(celltype_cfg)
    for type_name in selected_types:
        mask = masks.get(type_name)
        if mask is None or not np.any(mask):
            continue
        color = name_to_color.get(type_name, boundary_color) if use_type_colors else boundary_color
        _plot_mask_contours(
            ax,
            mask=mask,
            color=color,
            line_width=line_width,
            line_style=line_style,
            contour_downsample=contour_downsample,
        )

    if title:
        add_colored_type_text(
            ax,
            [name for name in ct_names if name in selected_types],
            [name_to_color[name] for name in ct_names if name in selected_types],
            title=title,
            fontsize=14,
        )
    else:
        add_colored_type_text(
            ax,
            [name for name in ct_names if name in selected_types],
            [name_to_color[name] for name in ct_names if name in selected_types],
            fontsize=14,
        )
    add_scalebar_20um(ax, celltype_mask.shape, float(pixel_size_um[0]), bar_um=20.0, color="white", lw=4, pad_frac=0.05)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def make_roi_comparison_figure(
    overlay_rgb: np.ndarray,
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    roi_masks: Dict[str, np.ndarray],
    selected_types: Sequence[str],
    pixel_size_um: Tuple[float, float],
    title: str | None = None,
) -> plt.Figure:
    display_mask_rgb = make_celltype_mask_rgb(celltype_mask, celltype_cfg, selected_types=selected_types)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5))
    axes[0].imshow(np.clip(overlay_rgb, 0.0, 1.0), origin="upper")
    axes[1].imshow(np.clip(display_mask_rgb, 0.0, 1.0), origin="upper")
    for ax, panel_title in zip(axes, ["Overlay preview", "Cell-type mask"]):
        axis_off(ax)
        ax.text(
            0.985,
            0.985,
            panel_title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.30), edgecolor="none"),
        )
        add_scalebar_20um(ax, celltype_mask.shape, float(pixel_size_um[0]), bar_um=20.0, color="white", lw=4, pad_frac=0.05)

    roi_palette = ["#ffcc00", "#00ffff", "#ff66ff", "#ffffff", "#ff8800", "#66ff66", "#66a3ff"]
    for idx, (roi_name, mask) in enumerate(roi_masks.items()):
        color = roi_palette[idx % len(roi_palette)]
        for ax in axes:
            _plot_mask_contours(ax, mask=mask, color=color, line_width=2.0, line_style="-", contour_downsample=1)
            ax.text(
                0.02,
                0.98 - idx * 0.06,
                roi_name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=(0, 0, 0, 0.22), edgecolor="none"),
            )

    if title:
        fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.02)
    return fig


def _prepare_cell_positions(df_cells: pd.DataFrame, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    height, width = shape
    df_base = df_cells[["label", "celltype", "centroid_x_px", "centroid_y_px"]].copy()
    cy = np.clip(np.rint(df_base["centroid_y_px"].to_numpy(float)).astype(int), 0, height - 1)
    cx = np.clip(np.rint(df_base["centroid_x_px"].to_numpy(float)).astype(int), 0, width - 1)
    return cy, cx, df_base


def summarize_cells_in_named_masks(
    df_cells: pd.DataFrame,
    masks: Dict[str, np.ndarray],
    inside_name_fn=None,
    outside_label: str = "outside",
    mask_label_col: str = "region_name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not masks:
        empty = pd.DataFrame(columns=["label", "celltype", "centroid_x_px", "centroid_y_px", mask_label_col, "inside_region", "region"])
        empty_counts = pd.DataFrame(columns=[mask_label_col, "region", "celltype", "count"])
        return empty, empty_counts

    first_mask = next(iter(masks.values()))
    cy_all, cx_all, df_base = _prepare_cell_positions(df_cells, first_mask.shape)
    rows: List[pd.DataFrame] = []
    for mask_name, mask in masks.items():
        inside = mask[cy_all, cx_all] if np.any(mask) else np.zeros_like(cy_all, dtype=bool)
        tmp = df_base.copy()
        tmp[mask_label_col] = str(mask_name)
        tmp["inside_region"] = inside.astype(bool)
        inside_label = inside_name_fn(mask_name) if callable(inside_name_fn) else str(mask_name)
        tmp["region"] = np.where(inside, inside_label, outside_label)
        rows.append(tmp)
    df_assign = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    counts_by_region = (
        df_assign.groupby([mask_label_col, "region", "celltype"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values([mask_label_col, "region", "celltype"])
        .reset_index(drop=True)
    ) if not df_assign.empty else pd.DataFrame(columns=[mask_label_col, "region", "celltype", "count"])
    return df_assign, counts_by_region


def summarize_mask_areas(
    masks: Dict[str, np.ndarray],
    pixel_size_um: Tuple[float, float],
    label_col: str = "region_name",
) -> pd.DataFrame:
    if not masks:
        return pd.DataFrame(columns=[label_col, "area_px", "area_um2", "total_field_area_px", "total_field_area_um2", "area_fraction"])
    first_mask = next(iter(masks.values()))
    total_field_area_px = int(first_mask.shape[0] * first_mask.shape[1])
    px_area_um2 = float(pixel_size_um[0]) * float(pixel_size_um[1])
    total_field_area_um2 = total_field_area_px * px_area_um2
    rows: List[Dict[str, Any]] = []
    for name, mask in masks.items():
        area_px = int(np.count_nonzero(mask))
        rows.append(
            {
                label_col: str(name),
                "area_px": area_px,
                "area_um2": float(area_px * px_area_um2),
                "total_field_area_px": total_field_area_px,
                "total_field_area_um2": float(total_field_area_um2),
                "area_fraction": float(area_px / total_field_area_px) if total_field_area_px > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(label_col).reset_index(drop=True)


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


def _save_masks_as_uint8(masks: Dict[str, np.ndarray], save_dir: Path, suffix_template: str) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for name, mask in masks.items():
        path = save_dir / suffix_template.format(name=safe_name(name, "region"))
        save_uint8_tiff(path, mask.astype(np.uint8))
        paths[name] = path
    return paths


def _boundary_registry_path(save_dir: Path) -> Path:
    return Path(save_dir) / "boundary_mask_registry.json"


def _load_boundary_registry(save_dir: Path) -> List[Dict[str, Any]]:
    path = _boundary_registry_path(save_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []

    if isinstance(payload, dict):
        records = payload.get("entries", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []

    cleaned: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        rel_path = str(record.get("mask_path", "")).strip()
        if not rel_path:
            continue
        abs_path = Path(save_dir) / rel_path
        if not abs_path.exists():
            continue
        cleaned.append(
            {
                "mask_path": rel_path,
                "display_name": str(record.get("display_name") or Path(rel_path).stem),
                "source": str(record.get("source") or ""),
                "group_name": str(record.get("group_name") or ""),
                "mask_key": str(record.get("mask_key") or ""),
            }
        )
    return cleaned


def _write_boundary_registry(save_dir: Path, records: Sequence[Dict[str, Any]]) -> Path:
    path = _boundary_registry_path(save_dir)
    write_json(path, {"entries": list(records)})
    return path


def _update_boundary_registry(
    save_dir: Path,
    entries: Sequence[Dict[str, Any]],
    replace_group_name: str | None = None,
    replace_source: str | None = None,
) -> Path:
    existing: Dict[str, Dict[str, Any]] = {}
    for record in _load_boundary_registry(save_dir):
        same_group = replace_group_name is None or str(record.get("group_name") or "") == str(replace_group_name)
        same_source = replace_source is None or str(record.get("source") or "") == str(replace_source)
        if (replace_group_name is not None or replace_source is not None) and same_group and same_source:
            continue
        existing[str(record.get("mask_path"))] = record
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rel_path = str(entry.get("mask_path", "")).strip()
        if not rel_path:
            continue
        payload = dict(entry)
        payload["mask_path"] = rel_path
        existing[rel_path] = payload
    records = sorted(
        existing.values(),
        key=lambda record: (
            str(record.get("source", "")),
            str(record.get("display_name", "")),
            str(record.get("mask_path", "")),
        ),
    )
    return _write_boundary_registry(save_dir, records)

def apply_boundary_edit_to_mask(
    original_mask: np.ndarray,
    edit_mask: np.ndarray,
    operation: str,
    connect_radius_px: int = 5,
) -> np.ndarray:
    original_mask = original_mask.astype(bool)
    edit_mask = edit_mask.astype(bool)
    if not np.any(edit_mask):
        return original_mask.copy()

    operation = str(operation).strip().lower()
    if operation == "include":
        updated = original_mask | ndi.binary_fill_holes(edit_mask)
    elif operation == "exclude":
        updated = original_mask & (~edit_mask)
    elif operation == "connect":
        radius = max(1, int(connect_radius_px))
        bridge = morphology.binary_dilation(edit_mask, morphology.disk(radius))
        updated = morphology.binary_closing(original_mask | bridge, morphology.disk(radius))
    else:
        updated = original_mask.copy()

    updated = ndi.binary_fill_holes(updated)
    return updated.astype(bool)


def _save_region_like_outputs(
    save_dir: Path,
    base_prefix: str,
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    masks: Dict[str, np.ndarray],
    selected_types: Sequence[str],
    df_cells: pd.DataFrame,
    pixel_size_um: Tuple[float, float],
    title: str,
    params_payload: Dict[str, Any],
    line_width: float = 2.0,
    line_style: str = "-",
    boundary_color: str = "#a1d99b",
    use_type_colors: bool = True,
    contour_downsample: int = 1,
    registry_mask_names: Sequence[str] | None = None,
    replace_registry_group_name: str | None = None,
    replace_registry_source: str | None = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    base_prefix = safe_name(base_prefix, "region")
    figure = make_region_overlay_figure(
        celltype_mask=celltype_mask,
        celltype_cfg=celltype_cfg,
        masks=masks,
        selected_types=selected_types,
        pixel_size_um=pixel_size_um,
        title=title,
        line_width=line_width,
        line_style=line_style,
        boundary_color=boundary_color,
        use_type_colors=use_type_colors,
        contour_downsample=contour_downsample,
    )
    assignments_df, counts_by_region = summarize_cells_in_named_masks(
        df_cells=df_cells,
        masks=masks,
        inside_name_fn=lambda name: f"{name}_region",
        outside_label="adjacent_region",
        mask_label_col="boundary_type",
    )
    area_summary = summarize_mask_areas(masks, pixel_size_um, label_col="boundary_type")
    if not area_summary.empty:
        inside_counts = (
            assignments_df[assignments_df["inside_region"]]
            .groupby("boundary_type")
            .size()
            .rename("n_cells_inside")
            .reset_index()
        )
        area_summary = area_summary.merge(inside_counts, on="boundary_type", how="left").fillna({"n_cells_inside": 0})
        area_summary["n_cells_inside"] = area_summary["n_cells_inside"].astype(int)

    overlay_svg = save_dir / f"celltypes_with_boundaries__{base_prefix}.svg"
    overlay_png = save_dir / f"celltypes_with_boundaries__{base_prefix}.png"
    overlay_tiff = save_dir / f"celltypes_with_boundaries__{base_prefix}.tiff"
    counts_csv = save_dir / f"celltype_counts_by_region__{base_prefix}.csv"
    assign_csv = save_dir / f"cell_region_assignments__{base_prefix}.csv"
    area_csv = save_dir / f"region_area_summary__{base_prefix}.csv"
    params_json = save_dir / f"region_params__{base_prefix}.json"
    mask_paths = {}
    registry_path = _boundary_registry_path(save_dir)

    display_name_map = params_payload.get("boundary_display_names", {})
    if not isinstance(display_name_map, dict):
        display_name_map = {}
    workflow_name = str(params_payload.get("workflow") or "computational_roi_identification")
    registry_mask_name_set = (
        {str(name) for name in registry_mask_names}
        if registry_mask_names is not None
        else {str(name) for name in masks.keys()}
    )

    if save_outputs:
        figure.savefig(overlay_svg, dpi=600, bbox_inches="tight", pad_inches=0)
        figure.savefig(overlay_png, dpi=300, bbox_inches="tight", pad_inches=0)
        figure.savefig(overlay_tiff, dpi=600, bbox_inches="tight", pad_inches=0)
        counts_by_region.to_csv(counts_csv, index=False)
        assignments_df.to_csv(assign_csv, index=False)
        area_summary.to_csv(area_csv, index=False)
        write_json(params_json, params_payload)
        mask_paths = _save_masks_as_uint8(masks, save_dir, suffix_template="{name}__" + base_prefix + "_region_mask_uint8.tiff")

        registry_entries = []
        for mask_name, path in mask_paths.items():
            if str(mask_name) not in registry_mask_name_set:
                continue
            display_name = str(display_name_map.get(mask_name, mask_name)).strip() or str(mask_name)
            registry_entries.append(
                {
                    "mask_path": str(Path(path).relative_to(save_dir)),
                    "display_name": display_name,
                    "source": workflow_name,
                    "group_name": str(base_prefix),
                    "mask_key": str(mask_name),
                }
            )
        registry_path = _update_boundary_registry(
            save_dir,
            registry_entries,
            replace_group_name=replace_registry_group_name,
            replace_source=replace_registry_source,
        )

    return {
        "masks": {name: mask.astype(bool) for name, mask in masks.items()},
        "figure": figure,
        "counts_by_region": counts_by_region,
        "assignments": assignments_df,
        "area_summary": area_summary,
        "saved_paths": {
            "overlay_svg": overlay_svg,
            "overlay_png": overlay_png,
            "overlay_tiff": overlay_tiff,
            "counts_csv": counts_csv,
            "assignments_csv": assign_csv,
            "area_summary_csv": area_csv,
            "params_json": params_json,
            "mask_paths": mask_paths,
            "boundary_registry_json": registry_path,
        },
    }

def save_adjusted_region_analysis(
    df_cells: pd.DataFrame,
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    adjusted_masks: Dict[str, np.ndarray],
    selected_types: Sequence[str],
    edit_meta: Dict[str, Any],
    edited_boundary_types: Sequence[str] | None = None,
    boundary_display_names: Dict[str, str] | None = None,
    line_width: float = 2.0,
    line_style: str = "-",
    boundary_color: str = "#a1d99b",
    use_type_colors: bool = True,
    contour_downsample: int = 1,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    base_prefix = "adjusted__" + "__".join(sorted(str(name) for name in adjusted_masks.keys())) if adjusted_masks else "adjusted"
    display_name_map = {str(name): str(name) for name in adjusted_masks.keys()}
    if isinstance(boundary_display_names, dict):
        for key, value in boundary_display_names.items():
            cleaned_value = str(value).strip()
            if cleaned_value:
                display_name_map[str(key)] = cleaned_value
    edited_mask_names = [
        str(name)
        for name in (edited_boundary_types or [])
        if str(name) in adjusted_masks
    ]
    if not edited_mask_names:
        target_type = str(edit_meta.get("target_type") or "").strip()
        if target_type and target_type in adjusted_masks:
            edited_mask_names = [target_type]
    params_payload = {
        "workflow": "manual_boundary_adjustment",
        "selected_types": list(selected_types),
        "edit_meta": edit_meta,
        "edited_boundary_types": list(edited_mask_names),
        "boundary_display_names": display_name_map,
        "line_width": float(line_width),
        "line_style": str(line_style),
        "boundary_color": str(boundary_color),
        "use_type_colors": bool(use_type_colors),
        "contour_downsample": int(contour_downsample),
    }
    return _save_region_like_outputs(
        save_dir=save_dir,
        base_prefix=base_prefix,
        celltype_mask=celltype_mask,
        celltype_cfg=celltype_cfg,
        masks=adjusted_masks,
        selected_types=selected_types,
        df_cells=df_cells,
        pixel_size_um=pixel_size_um,
        title="Adjusted regions: " + ", ".join(selected_types),
        params_payload=params_payload,
        line_width=line_width,
        line_style=line_style,
        boundary_color=boundary_color,
        use_type_colors=use_type_colors,
        contour_downsample=contour_downsample,
        registry_mask_names=edited_mask_names,
        replace_registry_source="manual_boundary_adjustment",
        save_outputs=save_outputs,
    )

def save_manual_roi_analysis(
    df_cells: pd.DataFrame,
    celltype_mask: np.ndarray,
    celltype_cfg: Sequence[Dict[str, Any]],
    overlay_rgb: np.ndarray,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    roi_label_mask: np.ndarray,
    selected_types: Sequence[str],
    roi_source_panel: str,
    roi_custom_names: Sequence[str] | None = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    roi_ids = [int(v) for v in np.unique(roi_label_mask) if int(v) > 0]
    raw_custom_names = [str(name).strip() for name in (roi_custom_names or [])]

    used_names: set[str] = set()
    display_names: List[str] = []
    for idx, roi_id in enumerate(roi_ids, start=1):
        base_name = raw_custom_names[idx - 1] if idx - 1 < len(raw_custom_names) and raw_custom_names[idx - 1] else f"ROI_{idx:03d}"
        candidate = str(base_name).strip() or f"ROI_{idx:03d}"
        suffix = 2
        while candidate.lower() in used_names:
            candidate = f"{base_name}_{suffix:02d}"
            suffix += 1
        used_names.add(candidate.lower())
        display_names.append(candidate)

    roi_masks = {display_name: (roi_label_mask == roi_id) for display_name, roi_id in zip(display_names, roi_ids)}

    assignments_df, counts_by_roi = summarize_cells_in_named_masks(
        df_cells=df_cells,
        masks=roi_masks,
        inside_name_fn=lambda name: name,
        outside_label="outside_roi",
        mask_label_col="roi_name",
    )
    area_summary = summarize_mask_areas(roi_masks, pixel_size_um, label_col="roi_name")
    if not area_summary.empty:
        inside_counts = (
            assignments_df[assignments_df["inside_region"]]
            .groupby("roi_name")
            .size()
            .rename("n_cells_inside")
            .reset_index()
        )
        area_summary = area_summary.merge(inside_counts, on="roi_name", how="left").fillna({"n_cells_inside": 0})
        area_summary["n_cells_inside"] = area_summary["n_cells_inside"].astype(int)

    fig = make_roi_comparison_figure(
        overlay_rgb=overlay_rgb,
        celltype_mask=celltype_mask,
        celltype_cfg=celltype_cfg,
        roi_masks=roi_masks,
        selected_types=selected_types,
        pixel_size_um=pixel_size_um,
        title="Manual ROI selection",
    )

    base_prefix = safe_name("__".join(selected_types) if selected_types else "all_types", "manual_roi")
    overlay_svg = save_dir / f"manual_roi_comparison__{base_prefix}.svg"
    overlay_png = save_dir / f"manual_roi_comparison__{base_prefix}.png"
    overlay_tiff = save_dir / f"manual_roi_comparison__{base_prefix}.tiff"
    counts_csv = save_dir / f"celltype_counts_by_roi__{base_prefix}.csv"
    assign_csv = save_dir / f"cell_roi_assignments__{base_prefix}.csv"
    area_csv = save_dir / f"roi_area_summary__{base_prefix}.csv"
    params_json = save_dir / f"roi_params__{base_prefix}.json"
    roi_mask_tiff = save_dir / f"manual_roi_mask_uint16__{base_prefix}.tiff"
    roi_mask_paths: Dict[str, Path] = {}
    registry_path = _boundary_registry_path(save_dir)

    if save_outputs:
        fig.savefig(overlay_svg, dpi=600, bbox_inches="tight", pad_inches=0)
        fig.savefig(overlay_png, dpi=300, bbox_inches="tight", pad_inches=0)
        fig.savefig(overlay_tiff, dpi=600, bbox_inches="tight", pad_inches=0)
        counts_by_roi.to_csv(counts_csv, index=False)
        assignments_df.to_csv(assign_csv, index=False)
        area_summary.to_csv(area_csv, index=False)
        write_json(
            params_json,
            {
                "workflow": "manual_roi_selection",
                "selected_types": list(selected_types),
                "roi_source_panel": str(roi_source_panel),
                "n_rois": int(len(roi_masks)),
                "roi_display_names": list(roi_masks.keys()),
            },
        )
        save_uint16_tiff(roi_mask_tiff, roi_label_mask.astype(np.uint16))

        registry_entries = []
        for roi_name, mask in roi_masks.items():
            roi_mask_path = save_dir / f"manual_roi__{safe_name(roi_name, 'roi')}_region_mask_uint8.tiff"
            save_uint8_tiff(roi_mask_path, np.asarray(mask).astype(np.uint8))
            roi_mask_paths[roi_name] = roi_mask_path
            registry_entries.append(
                {
                    "mask_path": str(roi_mask_path.relative_to(save_dir)),
                    "display_name": str(roi_name),
                    "source": "manual_roi_selection",
                    "group_name": str(base_prefix),
                    "mask_key": str(roi_name),
                }
            )
        registry_path = _update_boundary_registry(
            save_dir,
            registry_entries,
            replace_group_name=str(base_prefix),
            replace_source="manual_roi_selection",
        )

    return {
        "roi_masks": roi_masks,
        "roi_label_mask": roi_label_mask.astype(np.uint16),
        "figure": fig,
        "counts_by_roi": counts_by_roi,
        "assignments": assignments_df,
        "area_summary": area_summary,
        "saved_paths": {
            "comparison_svg": overlay_svg,
            "comparison_png": overlay_png,
            "comparison_tiff": overlay_tiff,
            "counts_csv": counts_csv,
            "assignments_csv": assign_csv,
            "area_summary_csv": area_csv,
            "params_json": params_json,
            "roi_mask_tiff": roi_mask_tiff,
            "mask_paths": roi_mask_paths,
            "boundary_registry_json": registry_path,
        },
    }

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

    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    px_area_um2 = px_um_x * px_um_y

    required_cols = {"label", "celltype", "centroid_x_px", "centroid_y_px"}
    if not required_cols.issubset(set(df_cells.columns)):
        raise RuntimeError(f"df_cells must contain columns: {required_cols}")

    ct_names = [ct["name"] for ct in celltype_cfg]
    name_to_id = {ct_names[i]: (i + 1) for i in range(len(ct_names))}

    close_px = um_to_px_iso(params.close_um, pixel_size_um)
    dilate_px = um_to_px_iso(params.dilate_um, pixel_size_um)
    min_area_px = int(round(float(params.min_area_um2) / max(1e-12, px_area_um2)))
    min_cells = int(params.min_cells)

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

    result = _save_region_like_outputs(
        save_dir=save_dir,
        base_prefix="__".join(params.selected_types),
        celltype_mask=celltype_mask,
        celltype_cfg=celltype_cfg,
        masks=masks,
        selected_types=params.selected_types,
        df_cells=df_cells,
        pixel_size_um=pixel_size_um,
        title="Boundaries: " + ", ".join(params.selected_types),
        params_payload=params.to_dict(),
        line_width=float(params.line_width),
        line_style=str(params.line_style),
        boundary_color=str(params.boundary_color),
        use_type_colors=bool(params.use_type_colors),
        contour_downsample=int(params.contour_downsample),
        save_outputs=save_outputs,
    )
    result["celltype_mask"] = celltype_mask
    result["celltype_cfg"] = list(celltype_cfg)
    result["params_used"] = params.to_dict()
    return result


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
