from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from .io import load_any_tiff, load_text_grid, save_uint16_tiff, write_json
from .visualization import (
    add_colored_type_text,
    add_scalebar_20um,
    axis_off,
    build_celltype_cmap,
    COLOR_HEX_LIST,
    COMMON_FIRST,
)
from .io import valid_pixel_size, to_image


def guess_nuclear_channel(channels: Sequence[str]) -> str | None:
    if not channels:
        return None
    upper = {channel.upper(): channel for channel in channels}
    for key in ["DAPI", "HOECHST", "NUCLEUS", "NUCLEAR"]:
        if key in upper:
            return upper[key]
    for channel in channels:
        upper_channel = channel.upper()
        if ("DAPI" in upper_channel) or ("HOECHST" in upper_channel) or ("NUC" in upper_channel):
            return channel
    return None


def marker_choices_for_ui(channel_names: Sequence[str]) -> List[str]:
    nuc_guess = guess_nuclear_channel(channel_names)
    marker_choices = ["nucleus"] + ([c for c in channel_names if c != nuc_guess] if nuc_guess else list(channel_names))
    seen = set()
    return [m for m in marker_choices if not (m in seen or seen.add(m))]


def safe_token(name: str) -> str:
    if str(name).strip().lower() == "nucleus":
        return "NUCLEUS"
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    return token.upper() if token else "MARKER"


def token_mapping_for_ui(channel_names: Sequence[str]) -> Dict[str, str]:
    return {name: safe_token(name) for name in marker_choices_for_ui(channel_names)}


def safe_key(name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    return key.upper() if key else "MARKER"


NUC_KEY = "NUCLEUS"


def marker_name_to_key(marker_name: str) -> str:
    upper_name = (marker_name or "").strip().upper()
    if upper_name in {"NUCLEUS", "NUCLEAR", "NUC"}:
        return NUC_KEY
    if "DAPI" in upper_name or "HOECHST" in upper_name:
        return NUC_KEY
    return safe_key(marker_name)


def normalize_expr(expr: str) -> str:
    expr_norm = (expr or "").strip()
    expr_norm = re.sub(r"\bAND\b", "and", expr_norm, flags=re.I)
    expr_norm = re.sub(r"\bOR\b", "or", expr_norm, flags=re.I)
    expr_norm = re.sub(r"\bNOT\b", "not", expr_norm, flags=re.I)
    return expr_norm


def default_celltype(name: str, color_hex: str) -> Dict[str, Any]:
    return {
        "name": name,
        "color_hex": color_hex,
        "mode": "simple",
        "all_pos": [],
        "all_neg": [],
        "any_pos_groups": [],
    }


def save_celltype_config(celltype_cfg: Sequence[Dict[str, Any]], save_dir: Path) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "celltype_config.json"
    path.write_text(json.dumps(list(celltype_cfg), indent=2))
    return path


from dataclasses import asdict, dataclass, replace
from itertools import product
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class CelltypeAssignmentParams:
    r_voronoi_um: float = 3.0
    r_buffer_um: float = 2.0
    r_vote_um: float = 3.0
    tophat_r_um: float = 1.0
    gauss_sigma_um: float = 0.5
    thresh_mode: str = "global_otsu"
    min_pos_object_size_px: int = 9
    min_pos_pix: int = 5
    resolve_ambiguous: bool = True
    ambiguous_min_probability: float = 0.60
    ambiguous_min_gap: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CELLTYPE_PARAM_ORDER = [
    "r_voronoi_um",
    "r_buffer_um",
    "r_vote_um",
    "tophat_r_um",
    "gauss_sigma_um",
    "thresh_mode",
    "min_pos_object_size_px",
    "min_pos_pix",
]

CELLTYPE_PARAM_LABELS = {
    "r_voronoi_um": "R_VORONOI_UM",
    "r_buffer_um": "R_BUFFER_UM",
    "r_vote_um": "R_VOTE_UM",
    "tophat_r_um": "TOPHAT_R_UM",
    "gauss_sigma_um": "GAUSS_SIGMA_UM",
    "thresh_mode": "THRESH_MODE",
    "min_pos_object_size_px": "MIN_POS_OBJECT_SIZE_PX",
    "min_pos_pix": "MIN_POS_PIX",
}


def make_celltype_assignment_parameter_sweep_figure(
    df_results: pd.DataFrame,
    count_columns: Sequence[str],
    celltype_cfg: Sequence[Dict[str, Any]] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if df_results is None or len(df_results) == 0:
        ax.text(0.5, 0.5, "No parameter-scan results", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ok = df_results.copy()
    if "error" in ok.columns:
        ok = ok[ok["error"].fillna("") == ""].copy()

    if ok.empty:
        ax.text(0.5, 0.5, "No successful parameter-scan combinations", ha="center", va="center")
        ax.set_axis_off()
        return fig

    color_map: Dict[str, str] = {}
    if celltype_cfg is not None:
        for ct in celltype_cfg:
            name = str(ct.get("name", "")).strip()
            color_hex = str(ct.get("color_hex", "")).strip()
            if name and color_hex:
                color_map[name] = color_hex
    color_map.setdefault("Unassigned", "#808080")
    color_map.setdefault("Ambiguous", "#202020")

    x = ok["combo_index"].to_numpy()
    for col in count_columns:
        if col not in ok.columns:
            continue
        label = col.replace("count::", "")
        line_kwargs = {
            "marker": "o",
            "linewidth": 1.6,
            "markersize": 3.5,
            "label": label,
        }
        if label in color_map:
            line_kwargs["color"] = color_map[label]
        if label == "Unassigned":
            line_kwargs["linestyle"] = ":"
        elif label == "Ambiguous":
            line_kwargs["linestyle"] = "--"
        ax.plot(x, ok[col].to_numpy(), **line_kwargs)

    ax.set_xlabel("Combination index")
    ax.set_ylabel("Detected cells")
    ax.set_title("Cell-type counts across parameter combinations")
    ax.grid(alpha=0.25)
    if len(count_columns) <= 10:
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig



def _coerce_assignment_params(
    params: CelltypeAssignmentParams | Dict[str, Any] | None,
) -> CelltypeAssignmentParams:
    if params is None:
        return CelltypeAssignmentParams()
    if isinstance(params, CelltypeAssignmentParams):
        return params
    if isinstance(params, dict):
        payload = {}
        allowed_fields = set(CELLTYPE_PARAM_ORDER) | {"resolve_ambiguous", "ambiguous_min_probability", "ambiguous_min_gap"}
        for field in allowed_fields:
            if field in params:
                payload[field] = params[field]
        if "thresh_mode" in payload:
            payload["thresh_mode"] = str(payload["thresh_mode"])
        if "min_pos_object_size_px" in payload:
            payload["min_pos_object_size_px"] = max(0, int(payload["min_pos_object_size_px"]))
        if "min_pos_pix" in payload:
            payload["min_pos_pix"] = max(0, int(payload["min_pos_pix"]))
        for field in ["r_voronoi_um", "r_buffer_um", "r_vote_um", "tophat_r_um", "gauss_sigma_um", "ambiguous_min_probability", "ambiguous_min_gap"]:
            if field in payload:
                payload[field] = float(payload[field])
        if "resolve_ambiguous" in payload:
            payload["resolve_ambiguous"] = bool(payload["resolve_ambiguous"])
        return CelltypeAssignmentParams(**payload)
    raise TypeError(f"Unsupported assignment params type: {type(params)!r}")

def _run_celltype_assignment_impl(
    folder: Path,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    image_id: str,
    channels_cfg: Sequence[Dict[str, Any]],
    celltype_cfg: Sequence[Dict[str, Any]],
    labels: np.ndarray | None = None,
    df_pixels: pd.DataFrame | None = None,
    shapes: Dict[Tuple[str, str], Tuple[int, int]] | None = None,
    params: CelltypeAssignmentParams | Dict[str, Any] | None = None,
    save_outputs: bool = True,
    make_figures: bool = True,
    native_threads: int | None = None,
    support_workers: int | None = None,
) -> Dict[str, Any]:
    if not celltype_cfg:
        raise RuntimeError("CELLTYPE_CFG is empty.")

    params = _coerce_assignment_params(params)
    folder = Path(folder)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not valid_pixel_size(pixel_size_um):
        raise RuntimeError("PIXEL_SIZE_UM missing/invalid. Please provide valid x/y pixel sizes.")

    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    px_area_um2 = px_um_x * px_um_y

    cpu = os.cpu_count() or 1
    target_threads = int(native_threads) if native_threads is not None else int(os.environ.get("OMP_NUM_THREADS", max(1, cpu - 1)))
    target_threads = max(1, target_threads)
    numba_threads = min(target_threads, 24)
    support_n_jobs = max(1, int(support_workers if support_workers is not None else target_threads))

    if labels is None:
        label_path = save_dir / "nuclei_labels_uint16.tiff"
        if not label_path.exists():
            raise RuntimeError("Missing nuclei labels. Run nuclei segmentation first.")
        labels = load_any_tiff(label_path).astype(np.int32)

    if labels.ndim != 2:
        raise RuntimeError(f"Nuclei labels must be a 2D label image, got shape={labels.shape}")

    h, w = labels.shape
    n_labels = int(labels.max())
    if n_labels <= 0:
        raise RuntimeError("No nuclei labels were found in the current label mask.")

    ch2file = {c["channel"]: (folder / c["file"]) for c in channels_cfg}
    channel_names = list(ch2file.keys())
    nuc_channel = guess_nuclear_channel(channel_names)

    def get_channel_image(channel_name: str) -> np.ndarray:
        if df_pixels is not None and shapes is not None:
            try:
                return to_image(df_pixels, shapes, image_id, channel_name).astype(np.float32)
            except Exception:
                pass
        path = ch2file.get(channel_name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Channel {channel_name!r} file not found in CFG/Folder.")
        return load_text_grid(path).astype(np.float32, copy=False)

    def um_to_px_iso(value_um: float) -> int:
        scale = np.sqrt(max(1e-12, px_um_x * px_um_y))
        return max(1, int(round(float(value_um) / scale)))

    r_voronoi_px = um_to_px_iso(params.r_voronoi_um)
    r_buffer_px = um_to_px_iso(params.r_buffer_um)
    r_vote_px = um_to_px_iso(params.r_vote_um)
    tophat_px = um_to_px_iso(params.tophat_r_um) if float(params.tophat_r_um) > 0 else 0
    gauss_sigma = float(params.gauss_sigma_um) / max(1e-12, np.sqrt(px_um_x * px_um_y))
    gauss_sigma = max(0.0, gauss_sigma)
    thresh_mode = str(params.thresh_mode)
    min_pos_pix = max(0, int(params.min_pos_pix))
    min_pos_object_size_px = max(0, int(params.min_pos_object_size_px))

    outside = labels == 0
    dist_outside, idxs = ndi.distance_transform_edt(outside, return_indices=True)
    iy, ix = idxs
    nearest_label_map = labels[iy, ix]
    voronoi_band = outside & (dist_outside <= r_voronoi_px)

    lab2 = segmentation.expand_labels(labels, distance=r_buffer_px)
    boundaries_thick = segmentation.find_boundaries(lab2, mode="thick")
    buffer_zone = morphology.binary_dilation(boundaries_thick, morphology.disk(max(1, r_buffer_px // 2)))
    buffer_zone &= voronoi_band

    owner_map = labels.copy()
    owner_map[voronoi_band] = nearest_label_map[voronoi_band]

    def disk_offsets(radius: int) -> Tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        mask = (yy * yy + xx * xx) <= radius * radius
        return yy[mask].astype(np.int32), xx[mask].astype(np.int32)

    vote_dys, vote_dxs = disk_offsets(r_vote_px)
    cand_dys, cand_dxs = disk_offsets(r_buffer_px)

    try:
        from numba import njit, prange, set_num_threads

        try:
            set_num_threads(numba_threads)
        except Exception:
            pass
        numba_ok = True
    except Exception:
        numba_ok = False

    if numba_ok:

        @njit(parallel=True, fastmath=True)
        def resolve_buffer_pixels(
            buf_ys,
            buf_xs,
            img_norm,
            owner_map_local,
            lab2_local,
            nearest_label_map_local,
            vote_dys_local,
            vote_dxs_local,
            cand_dys_local,
            cand_dxs_local,
            h_local,
            w_local,
        ):
            max_cands = 8
            out = np.zeros(buf_ys.shape[0], np.int32)
            for i in prange(buf_ys.shape[0]):
                y = buf_ys[i]
                x = buf_xs[i]
                cand_labels = np.zeros(max_cands, np.int32)
                n_cand = 0
                for k in range(cand_dys_local.shape[0]):
                    yy = y + cand_dys_local[k]
                    xx = x + cand_dxs_local[k]
                    if 0 <= yy < h_local and 0 <= xx < w_local:
                        lbl = lab2_local[yy, xx]
                        if lbl > 0:
                            seen = False
                            for j in range(n_cand):
                                if cand_labels[j] == lbl:
                                    seen = True
                                    break
                            if (not seen) and (n_cand < max_cands):
                                cand_labels[n_cand] = lbl
                                n_cand += 1
                if n_cand == 0:
                    out[i] = nearest_label_map_local[y, x]
                    continue
                best_lbl = 0
                best_vote = -1.0
                for j in range(n_cand):
                    lbl = cand_labels[j]
                    vote = 0.0
                    for k in range(vote_dys_local.shape[0]):
                        yy = y + vote_dys_local[k]
                        xx = x + vote_dxs_local[k]
                        if 0 <= yy < h_local and 0 <= xx < w_local:
                            if owner_map_local[yy, xx] == lbl:
                                vote += img_norm[yy, xx]
                    if vote > best_vote:
                        best_vote = vote
                        best_lbl = lbl
                if best_lbl == 0:
                    best_lbl = nearest_label_map_local[y, x]
                out[i] = best_lbl
            return out

    else:

        def resolve_buffer_pixels(
            buf_ys,
            buf_xs,
            img_norm,
            owner_map_local,
            lab2_local,
            nearest_label_map_local,
            vote_dys_local,
            vote_dxs_local,
            cand_dys_local,
            cand_dxs_local,
            h_local,
            w_local,
        ):
            out = np.zeros(buf_ys.shape[0], np.int32)
            for i, (y, x) in enumerate(zip(buf_ys, buf_xs)):
                candidates = set()
                for dy, dx in zip(cand_dys_local, cand_dxs_local):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h_local and 0 <= xx < w_local:
                        lbl = lab2_local[yy, xx]
                        if lbl > 0:
                            candidates.add(int(lbl))
                if not candidates:
                    out[i] = int(nearest_label_map_local[y, x])
                    continue
                best_lbl, best_vote = 0, -1.0
                for lbl in candidates:
                    vote = 0.0
                    for dy, dx in zip(vote_dys_local, vote_dxs_local):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h_local and 0 <= xx < w_local and owner_map_local[yy, xx] == lbl:
                            vote += float(img_norm[yy, xx])
                    if vote > best_vote:
                        best_vote = vote
                        best_lbl = lbl
                out[i] = best_lbl or int(nearest_label_map_local[y, x])
            return out

    def preprocess_marker(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32, copy=False)
        lo, hi = np.nanpercentile(img, [1, 99.8])
        norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        if tophat_px > 0:
            norm = morphology.white_tophat(norm, footprint=morphology.disk(tophat_px))
        if gauss_sigma > 0:
            norm = filters.gaussian(norm, sigma=gauss_sigma, preserve_range=True)
        return norm

    def marker_positive_mask(img_norm: np.ndarray) -> Tuple[np.ndarray, float]:
        if thresh_mode == "global_otsu":
            thr = filters.threshold_otsu(img_norm)
        elif thresh_mode == "yen":
            thr = filters.threshold_yen(img_norm)
        elif thresh_mode == "triangle":
            thr = filters.threshold_triangle(img_norm)
        else:
            thr = filters.threshold_otsu(img_norm)
        pos = img_norm > thr
        if min_pos_object_size_px > 1:
            pos = morphology.remove_small_objects(pos, min_size=min_pos_object_size_px)
        return pos, float(thr)

    assign_channels = [ch for ch in channel_names if (nuc_channel is None or ch != nuc_channel)]
    assign_maps: Dict[str, np.ndarray] = {}
    df_stats_list: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, Any]] = []

    def assign_marker_fast(img: np.ndarray, marker_display_name: str):
        marker_key = marker_name_to_key(marker_display_name)
        if marker_key == NUC_KEY:
            raise ValueError("Do not run marker assignment for nucleus (built-in).")

        img_norm = preprocess_marker(img)
        pos_mask, thr = marker_positive_mask(img_norm)
        threshold_rows.append(
            {
                "marker_display_name": marker_display_name,
                "marker_key": marker_key,
                "threshold": float(thr),
                "positive_pixels": int(pos_mask.sum()),
            }
        )

        assign_map = np.zeros_like(labels, dtype=np.uint16)

        inside_mask = (labels > 0) & pos_mask
        assign_map[inside_mask] = labels[inside_mask].astype(np.uint16)

        band_mask = pos_mask & voronoi_band
        nonbuf_mask = band_mask & (~buffer_zone)
        assign_map[nonbuf_mask] = nearest_label_map[nonbuf_mask].astype(np.uint16)

        by, bx = np.nonzero(pos_mask & buffer_zone)
        if by.size:
            picked = resolve_buffer_pixels(
                by.astype(np.int32),
                bx.astype(np.int32),
                img_norm,
                owner_map,
                lab2,
                nearest_label_map,
                vote_dys,
                vote_dxs,
                cand_dys,
                cand_dxs,
                h,
                w,
            )
            assign_map[by, bx] = picked.astype(np.uint16)

        flat_lab = assign_map.ravel().astype(np.int32)
        flat_val = img_norm.ravel()
        max_lab = int(flat_lab.max()) if flat_lab.size else 0
        pix_counts = np.bincount(flat_lab, minlength=max_lab + 1).astype(np.int64)
        val_sums = np.bincount(flat_lab, weights=flat_val, minlength=max_lab + 1)

        dfm = pd.DataFrame(
            {
                "label": np.arange(max_lab + 1, dtype=int),
                f"{marker_key}_pos_pix": pix_counts,
                f"{marker_key}_sum_intensity": val_sums,
            }
        )
        dfm = dfm[dfm["label"] > 0].reset_index(drop=True)

        if save_outputs:
            save_uint16_tiff(save_dir / f"marker_assign_{marker_key}_uint16.tiff", assign_map.astype(np.uint16))

        return marker_key, assign_map, dfm

    with threadpool_limits(limits=target_threads):
        for channel in assign_channels:
            marker_key = marker_name_to_key(channel)
            if marker_key == NUC_KEY:
                continue
            img = get_channel_image(channel)
            key, amap, dfm = assign_marker_fast(img, channel)
            assign_maps[key] = amap
            df_stats_list.append(dfm)

    def build_df_props_from_current_labels(labels_current: np.ndarray) -> pd.DataFrame:
        intensity_image = None
        if nuc_channel is not None:
            try:
                intensity_image = get_channel_image(nuc_channel)
            except Exception:
                intensity_image = None

        base_props = ("label", "area", "perimeter", "eccentricity", "solidity", "centroid", "bbox")
        if intensity_image is not None:
            props = measure.regionprops_table(
                labels_current,
                intensity_image=intensity_image,
                properties=base_props + ("mean_intensity", "max_intensity"),
            )
        else:
            props = measure.regionprops_table(labels_current, properties=base_props)

        out = pd.DataFrame(props)
        out.rename(
            columns={
                "centroid-0": "centroid_y_px",
                "centroid-1": "centroid_x_px",
                "bbox-0": "bbox_min_y_px",
                "bbox-1": "bbox_min_x_px",
                "bbox-2": "bbox_max_y_px",
                "bbox-3": "bbox_max_x_px",
            },
            inplace=True,
        )

        if not out.empty:
            out["label"] = out["label"].astype(int)
            out["centroid_x_um"] = out["centroid_x_px"].to_numpy(float) * px_um_x
            out["centroid_y_um"] = out["centroid_y_px"].to_numpy(float) * px_um_y
            out["area_um2"] = out["area"].to_numpy(float) * px_area_um2
            out["perimeter_um"] = out["perimeter"].to_numpy(float) * np.sqrt(max(1e-12, px_um_x * px_um_y))

        return out.sort_values("label").reset_index(drop=True)

    df_props = build_df_props_from_current_labels(labels)
    if df_props.empty:
        raise RuntimeError("No nuclei properties could be computed. Segmentation labels appear empty.")

    df_cells = df_props.copy()
    for dfm in df_stats_list:
        df_cells = df_cells.merge(dfm, on="label", how="left")
    df_cells.fillna(0, inplace=True)

    nuc_area = np.bincount(labels.ravel().astype(np.int64), minlength=n_labels + 1).astype(np.int64)
    lab_idx = df_cells["label"].to_numpy(np.int64)

    if len(lab_idx) == 0:
        raise RuntimeError("No nuclei labels were available for cell-type assignment.")
    if lab_idx.min() < 1 or lab_idx.max() > n_labels:
        raise RuntimeError(
            f"Label mismatch: df_cells has labels {lab_idx.min()}..{lab_idx.max()}, but current labels max is {n_labels}."
        )

    df_cells["NUCLEUS_pos_pix"] = nuc_area[lab_idx]
    df_cells["NUCLEUS_pos"] = df_cells["NUCLEUS_pos_pix"] > 0

    if "centroid_x_um" not in df_cells.columns:
        df_cells["centroid_x_um"] = df_cells["centroid_x_px"].to_numpy(float) * px_um_x
    if "centroid_y_um" not in df_cells.columns:
        df_cells["centroid_y_um"] = df_cells["centroid_y_px"].to_numpy(float) * px_um_y

    marker_keys = sorted(assign_maps.keys())
    for marker_key in marker_keys:
        marker_pix = df_cells.get(f"{marker_key}_pos_pix", 0)
        if min_pos_pix <= 0:
            df_cells[f"{marker_key}_pos"] = marker_pix > 0
        else:
            df_cells[f"{marker_key}_pos"] = marker_pix >= min_pos_pix

    def is_pos(row: pd.Series, marker_key: str) -> bool:
        if marker_key == NUC_KEY:
            return bool(row.get("NUCLEUS_pos", False))
        col = f"{marker_key}_pos"
        return bool(row.get(col, False))

    def match_simple(ct: Dict[str, Any], row: pd.Series) -> bool:
        all_pos = [marker_name_to_key(marker) for marker in ct.get("all_pos", [])]
        all_neg = [marker_name_to_key(marker) for marker in ct.get("all_neg", [])]
        any_groups = [[marker_name_to_key(marker) for marker in group] for group in ct.get("any_pos_groups", [])]

        if not all(is_pos(row, mk) for mk in all_pos):
            return False
        if not all((not is_pos(row, mk)) for mk in all_neg):
            return False
        for group in any_groups:
            if group and (not any(is_pos(row, mk) for mk in group)):
                return False
        return True


    compiled_expr: List[Any] = []
    compiled_expr_ast: List[Any] = []
    for ct in celltype_cfg:
        if ct.get("mode") == "expr":
            expr = normalize_expr(ct.get("expr", ""))
            try:
                compiled_expr.append(compile(expr, "<celltype_expr>", "eval") if expr else None)
                compiled_expr_ast.append(ast.parse(expr, mode="eval").body if expr else None)
            except Exception:
                compiled_expr.append(None)
                compiled_expr_ast.append(None)
        else:
            compiled_expr.append(None)
            compiled_expr_ast.append(None)

    env_keys = [NUC_KEY] + marker_keys

    def match_expr(ct_index: int, row: pd.Series) -> bool:
        code = compiled_expr[ct_index]
        if code is None:
            return False
        env = {key: is_pos(row, key) for key in env_keys}
        try:
            return bool(eval(code, {"__builtins__": {}}, env))
        except Exception:
            return False

    def marker_probability(row: pd.Series, marker_key: str) -> float:
        if marker_key == NUC_KEY:
            pix = float(row.get("NUCLEUS_pos_pix", 0))
            return 1.0 if pix > 0 else 0.0
        pix = float(row.get(f"{marker_key}_pos_pix", 0.0))
        intensity = float(row.get(f"{marker_key}_sum_intensity", 0.0))
        scale_pix = max(1.0, float(max(min_pos_pix, 1)))
        pix_prob = 1.0 - np.exp(-max(0.0, pix) / scale_pix)
        intensity_prob = 1.0 - np.exp(-max(0.0, intensity) / max(1.0, scale_pix / 2.0))
        return float(np.clip(0.65 * pix_prob + 0.35 * intensity_prob, 0.0, 1.0))

    def negative_probability(row: pd.Series, marker_key: str) -> float:
        return float(np.clip(1.0 - marker_probability(row, marker_key), 0.0, 1.0))

    def geometric_mean_prob(values: Sequence[float]) -> float:
        vals = [float(np.clip(v, 1e-6, 1.0)) for v in values if v is not None]
        if not vals:
            return 0.5
        return float(np.exp(np.mean(np.log(vals))))

    def score_simple_probability(ct: Dict[str, Any], row: pd.Series) -> float:
        all_pos = [marker_name_to_key(marker) for marker in ct.get("all_pos", [])]
        all_neg = [marker_name_to_key(marker) for marker in ct.get("all_neg", [])]
        any_groups = [[marker_name_to_key(marker) for marker in group] for group in ct.get("any_pos_groups", [])]

        terms: List[float] = []
        for mk in all_pos:
            terms.append(marker_probability(row, mk))
        for mk in all_neg:
            terms.append(negative_probability(row, mk))
        for group in any_groups:
            if group:
                terms.append(max([marker_probability(row, mk) for mk in group] or [0.0]))
        return geometric_mean_prob(terms)

    def eval_probability_ast(node: Any, row: pd.Series) -> float:
        if node is None:
            return 0.0
        if isinstance(node, ast.Name):
            return marker_probability(row, marker_name_to_key(node.id))
        if isinstance(node, ast.Constant):
            return 1.0 if bool(node.value) else 0.0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            return float(np.clip(1.0 - eval_probability_ast(node.operand, row), 0.0, 1.0))
        if isinstance(node, ast.BoolOp):
            values = [eval_probability_ast(v, row) for v in node.values]
            if isinstance(node.op, ast.And):
                return geometric_mean_prob(values)
            if isinstance(node.op, ast.Or):
                return float(np.clip(1.0 - np.prod([1.0 - np.clip(v, 0.0, 1.0) for v in values]), 0.0, 1.0))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            return geometric_mean_prob([eval_probability_ast(node.left, row), eval_probability_ast(node.right, row)])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            lv = eval_probability_ast(node.left, row)
            rv = eval_probability_ast(node.right, row)
            return float(np.clip(1.0 - (1.0 - lv) * (1.0 - rv), 0.0, 1.0))
        return 0.0

    def score_expr_probability(ct_index: int, row: pd.Series) -> float:
        return float(np.clip(eval_probability_ast(compiled_expr_ast[ct_index], row), 0.0, 1.0))

    k_types = len(celltype_cfg)
    celltype_id = np.zeros(len(df_cells), dtype=np.uint16)
    celltype_name = np.array(["Unassigned"] * len(df_cells), dtype=object)
    matched_celltypes: List[str] = []
    n_matched_celltypes = np.zeros(len(df_cells), dtype=np.int32)
    ambiguous_best_type = np.array([""] * len(df_cells), dtype=object)
    ambiguous_best_probability = np.zeros(len(df_cells), dtype=float)
    ambiguous_second_probability = np.zeros(len(df_cells), dtype=float)
    ambiguous_probability_gap = np.zeros(len(df_cells), dtype=float)
    ambiguous_candidate_probabilities: List[str] = []

    for i in range(len(df_cells)):
        row = df_cells.iloc[i]
        matches: List[tuple[int, str, int]] = []
        for k, ct in enumerate(celltype_cfg, start=1):
            ct_index = k - 1
            ok = match_simple(ct, row) if ct.get("mode") == "simple" else match_expr(ct_index, row)
            if ok:
                matches.append((k, ct["name"], ct_index))

        n_matches = len(matches)
        n_matched_celltypes[i] = n_matches
        matched_celltypes.append("|".join(name for _, name, _ in matches))

        if n_matches == 1:
            celltype_id[i] = matches[0][0]
            celltype_name[i] = matches[0][1]
            ambiguous_candidate_probabilities.append("")
        elif n_matches == 0:
            celltype_id[i] = 0
            celltype_name[i] = "Unassigned"
            ambiguous_candidate_probabilities.append("")
        else:
            candidate_scores: List[tuple[int, str, float]] = []
            for k, name, ct_index in matches:
                ct = celltype_cfg[ct_index]
                score = score_simple_probability(ct, row) if ct.get("mode") == "simple" else score_expr_probability(ct_index, row)
                candidate_scores.append((k, name, float(max(score, 1e-6))))
            total_score = float(sum(score for _, _, score in candidate_scores))
            if total_score <= 0:
                probabilities = [(k, name, 1.0 / len(candidate_scores)) for k, name, _ in candidate_scores]
            else:
                probabilities = [(k, name, score / total_score) for k, name, score in candidate_scores]
            probabilities = sorted(probabilities, key=lambda item: item[2], reverse=True)
            best_k, best_name, best_prob = probabilities[0]
            second_prob = probabilities[1][2] if len(probabilities) > 1 else 0.0

            ambiguous_best_type[i] = best_name
            ambiguous_best_probability[i] = float(best_prob)
            ambiguous_second_probability[i] = float(second_prob)
            ambiguous_probability_gap[i] = float(best_prob - second_prob)
            ambiguous_candidate_probabilities.append(
                "; ".join([f"{name}={prob:.3f}" for _, name, prob in probabilities])
            )

            if params.resolve_ambiguous and best_prob >= float(params.ambiguous_min_probability) and (best_prob - second_prob) >= float(params.ambiguous_min_gap):
                celltype_id[i] = int(best_k)
                celltype_name[i] = best_name
            else:
                celltype_id[i] = 0
                celltype_name[i] = "Ambiguous"

    df_cells["celltype_id"] = celltype_id.astype(int)
    df_cells["celltype"] = celltype_name
    df_cells["matched_celltypes"] = matched_celltypes
    df_cells["n_matched_celltypes"] = n_matched_celltypes.astype(int)
    df_cells["ambiguous_best_type"] = ambiguous_best_type
    df_cells["ambiguous_best_probability"] = ambiguous_best_probability.astype(float)
    df_cells["ambiguous_second_probability"] = ambiguous_second_probability.astype(float)
    df_cells["ambiguous_probability_gap"] = ambiguous_probability_gap.astype(float)
    df_cells["ambiguous_candidate_probabilities"] = ambiguous_candidate_probabilities

    celltype_id_by_label = np.zeros(n_labels + 1, dtype=np.uint16)
    lab_ct = df_cells[["label", "celltype_id"]].to_numpy()
    celltype_id_by_label[lab_ct[:, 0].astype(int)] = lab_ct[:, 1].astype(np.uint16)

    def type_has_non_nucleus(ct: Dict[str, Any]) -> bool:
        if ct.get("mode") == "simple":
            keys: List[str] = []
            keys += [marker_name_to_key(m) for m in ct.get("all_pos", [])]
            for group in ct.get("any_pos_groups", []):
                keys += [marker_name_to_key(m) for m in group]
            keys = [key for key in keys if key != NUC_KEY]
            return len(keys) > 0
        expr = normalize_expr(ct.get("expr", ""))
        toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr)
        toks = [tok for tok in toks if tok.lower() not in {"and", "or", "not", "true", "false"}]
        toks = [tok for tok in toks if tok.upper() != NUC_KEY]
        return len(toks) > 0

    type_has_marker = np.array([False] + [type_has_non_nucleus(ct) for ct in celltype_cfg], dtype=bool)
    celltype_cfg_by_name = {str(ct["name"]): ct for ct in celltype_cfg}

    def support_marker_keys_for_assigned_row(row: pd.Series) -> List[str]:
        ct_name = str(row.get("celltype", "") or "")
        ct = celltype_cfg_by_name.get(ct_name)
        if not ct:
            return []
        keys: List[str] = []
        if ct.get("mode") == "simple":
            for marker in ct.get("all_pos", []):
                mk = marker_name_to_key(marker)
                if mk != NUC_KEY and mk in assign_maps and is_pos(row, mk):
                    keys.append(mk)
            for group in ct.get("any_pos_groups", []):
                group_keys = [marker_name_to_key(marker) for marker in group]
                for mk in group_keys:
                    if mk != NUC_KEY and mk in assign_maps and is_pos(row, mk):
                        keys.append(mk)
        else:
            for mk in marker_keys:
                if mk != NUC_KEY and mk in assign_maps and is_pos(row, mk):
                    keys.append(mk)
        return list(dict.fromkeys(keys))

    label_rows = {int(row["label"]): row for _, row in df_cells.iterrows()}
    label_to_support_marker_keys: Dict[int, List[str]] = {}
    for row in df_cells.itertuples(index=False):
        label_id = int(getattr(row, "label"))
        ct_id = int(getattr(row, "celltype_id", 0))
        if ct_id <= 0:
            continue
        row_series = label_rows.get(label_id)
        if row_series is None:
            continue
        label_to_support_marker_keys[label_id] = support_marker_keys_for_assigned_row(row_series)

    slices_list = ndi.find_objects(labels)
    margin = int(max(r_voronoi_px, r_buffer_px) + 2)

    def expand_slice(slice_pair, margin_local: int, h_local: int, w_local: int):
        sy, sx = slice_pair
        y0 = max(0, sy.start - margin_local)
        y1 = min(h_local, sy.stop + margin_local)
        x0 = max(0, sx.start - margin_local)
        x1 = min(w_local, sx.stop + margin_local)
        return slice(y0, y1), slice(x0, x1)


    def _smooth_support_within_territory(
        support_seed: np.ndarray,
        nucleus_mask: np.ndarray,
        territory_mask: np.ndarray,
        distance_loc: np.ndarray,
        marker_present: bool,
    ) -> np.ndarray:
        support = (support_seed | nucleus_mask).astype(bool)
        support &= territory_mask
        if not np.any(support):
            return nucleus_mask.astype(bool)

        close_radius = max(1, min(6, int(round(max(1, r_buffer_px) / 2.0))))
        if marker_present:
            blur_sigma = max(0.8, min(2.5, max(1, r_buffer_px) / 2.5))
            soft = ndi.gaussian_filter(support.astype(float), sigma=blur_sigma)
            threshold = 0.18
            support = soft > threshold
            support = morphology.binary_closing(support, footprint=morphology.disk(close_radius))
            support = morphology.binary_opening(support, footprint=morphology.disk(1))
        else:
            halo = territory_mask & (distance_loc <= max(1, r_buffer_px))
            support |= halo
            soft = ndi.gaussian_filter(support.astype(float), sigma=max(0.8, min(1.8, max(1, r_buffer_px) / 3.0)))
            support = soft > 0.28
            support = morphology.binary_closing(support, footprint=morphology.disk(max(1, min(close_radius, 3))))

        support &= territory_mask
        support |= nucleus_mask
        support = ndi.binary_fill_holes(support)
        support &= territory_mask
        support |= nucleus_mask
        return support.astype(bool)

    def support_for_label(label_id: int):
        slc0 = slices_list[label_id - 1]
        if slc0 is None:
            return label_id, None, None
        slc = expand_slice(slc0, margin, h, w)
        lbl_loc = labels[slc]
        own_loc = owner_map[slc]
        dist_loc = dist_outside[slc]
        ct = int(celltype_id_by_label[label_id])

        if ct <= 0:
            return label_id, ct, None

        nucleus_mask = lbl_loc == label_id
        territory_mask = own_loc == label_id

        marker_keys = label_to_support_marker_keys.get(label_id, [])
        marker_support = np.zeros_like(nucleus_mask, dtype=bool)
        for mk in marker_keys:
            amap = assign_maps.get(mk)
            if amap is not None:
                marker_support |= (amap[slc] == label_id)

        if ct >= 1 and ct <= k_types and (not type_has_marker[ct]):
            marker_support |= territory_mask & (dist_loc <= max(1, r_buffer_px))

        support = _smooth_support_within_territory(
            support_seed=marker_support,
            nucleus_mask=nucleus_mask,
            territory_mask=territory_mask,
            distance_loc=dist_loc,
            marker_present=bool(np.any(marker_support)),
        )
        return label_id, ct, (slc, support)

    with threadpool_limits(limits=target_threads):
        results = Parallel(n_jobs=support_n_jobs, prefer="threads", batch_size=64)(
            delayed(support_for_label)(label_id) for label_id in range(1, n_labels + 1)
        )

    celltype_mask = np.zeros_like(labels, dtype=np.uint16)
    for label_id, ct, payload in results:
        if payload is None:
            continue
        slc, support = payload
        sub = celltype_mask[slc]
        sub[support] = np.uint16(ct)
        celltype_mask[slc] = sub

    counts = df_cells["celltype"].value_counts().rename_axis("celltype").reset_index(name="count")
    thresholds_df = pd.DataFrame(threshold_rows)

    panel_fig = None
    split_fig = None

    panel_svg = save_dir / "celltypes_panel.svg"
    panel_png = save_dir / "celltypes_panel.png"
    panel_tiff = save_dir / "celltypes_panel.tiff"
    split_svg = save_dir / "celltypes_split_panels.svg"
    split_png = save_dir / "celltypes_split_panels.png"
    split_tiff = save_dir / "celltypes_split_panels.tiff"

    if make_figures:
        ct_names = [ct["name"] for ct in celltype_cfg]
        ct_hex = [ct["color_hex"] for ct in celltype_cfg]
        cmap_ct = ListedColormap([(0, 0, 0)] + [mcolors.to_rgb(hx) for hx in ct_hex])

        nuc_norm = None
        if nuc_channel is not None:
            try:
                nuc_img = get_channel_image(nuc_channel)
                p1, p99 = np.nanpercentile(nuc_img, [1, 99.8])
                nuc_norm = np.clip((nuc_img - p1) / (p99 - p1 + 1e-6), 0, 1)
            except Exception:
                nuc_norm = None

        if nuc_norm is None:
            panel_fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.imshow(celltype_mask, cmap=cmap_ct, origin="upper", interpolation="nearest", vmin=0, vmax=k_types)
            axis_off(ax)
            add_scalebar_20um(ax, celltype_mask.shape, px_um_x, bar_um=20.0)
            add_colored_type_text(ax, ct_names, ct_hex)
            panel_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        else:
            panel_fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
            ax0.imshow(nuc_norm, cmap="gray", origin="upper")
            axis_off(ax0)
            add_scalebar_20um(ax0, nuc_norm.shape, px_um_x, bar_um=20.0)
            ax0.text(
                0.985,
                0.985,
                nuc_channel,
                transform=ax0.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.25), edgecolor="none"),
            )

            ax1.imshow(celltype_mask, cmap=cmap_ct, origin="upper", interpolation="nearest", vmin=0, vmax=k_types)
            axis_off(ax1)
            add_scalebar_20um(ax1, celltype_mask.shape, px_um_x, bar_um=20.0)
            add_colored_type_text(ax1, ct_names, ct_hex)
            panel_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)

        split_fig, axes2 = plt.subplots(1, k_types, figsize=(6 * max(1, k_types), 6))
        if k_types == 1:
            axes2 = [axes2]

        for i in range(1, k_types + 1):
            ax = axes2[i - 1]
            mask_i = (celltype_mask == i).astype(np.uint8)
            cmap_i = ListedColormap([(0, 0, 0), mcolors.to_rgb(ct_hex[i - 1])])
            ax.imshow(mask_i, cmap=cmap_i, origin="upper", interpolation="nearest", vmin=0, vmax=1)
            axis_off(ax)
            add_scalebar_20um(ax, mask_i.shape, px_um_x, bar_um=20.0)
            ax.text(
                0.985,
                0.985,
                ct_names[i - 1],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=14,
                fontweight="bold",
                color=ct_hex[i - 1],
                bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.25), edgecolor="none"),
            )

        split_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)

    if save_outputs:
        save_uint16_tiff(save_dir / "celltypes_mask_uint16.tiff", celltype_mask.astype(np.uint16))
        df_cells.to_csv(save_dir / "cells_summary.csv", index=False)
        counts.to_csv(save_dir / "celltype_counts.csv", index=False)
        thresholds_df.to_csv(save_dir / "marker_assignment_thresholds.csv", index=False)
        save_celltype_config(celltype_cfg, save_dir)
        if make_figures and panel_fig is not None and split_fig is not None:
            panel_fig.savefig(panel_svg, bbox_inches="tight", pad_inches=0)
            panel_fig.savefig(panel_png, dpi=300, bbox_inches="tight", pad_inches=0)
            panel_fig.savefig(panel_tiff, dpi=600, bbox_inches="tight", pad_inches=0)
            split_fig.savefig(split_svg, bbox_inches="tight", pad_inches=0)
            split_fig.savefig(split_png, dpi=300, bbox_inches="tight", pad_inches=0)
            split_fig.savefig(split_tiff, dpi=600, bbox_inches="tight", pad_inches=0)

    return {
        "df_cells": df_cells,
        "counts": counts,
        "celltype_mask": celltype_mask,
        "thresholds": thresholds_df,
        "nuc_channel": nuc_channel,
        "panel_figure": panel_fig,
        "split_figure": split_fig,
        "params_used": params.to_dict(),
        "saved_paths": {
            "celltype_mask_tiff": save_dir / "celltypes_mask_uint16.tiff",
            "cells_summary_csv": save_dir / "cells_summary.csv",
            "celltype_counts_csv": save_dir / "celltype_counts.csv",
            "marker_assignment_thresholds_csv": save_dir / "marker_assignment_thresholds.csv",
            "panel_svg": panel_svg,
            "panel_png": panel_png,
            "panel_tiff": panel_tiff,
            "split_svg": split_svg,
            "split_png": split_png,
            "split_tiff": split_tiff,
            "celltype_config_json": save_dir / "celltype_config.json",
        },
    }


def run_celltype_assignment(
    folder: Path,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    image_id: str,
    channels_cfg: Sequence[Dict[str, Any]],
    celltype_cfg: Sequence[Dict[str, Any]],
    labels: np.ndarray | None = None,
    df_pixels: pd.DataFrame | None = None,
    shapes: Dict[Tuple[str, str], Tuple[int, int]] | None = None,
    params: CelltypeAssignmentParams | Dict[str, Any] | None = None,
    save_outputs: bool = True,
    make_figures: bool = True,
    native_threads: int | None = None,
    support_workers: int | None = None,
) -> Dict[str, Any]:
    return _run_celltype_assignment_impl(
        folder=folder,
        save_dir=save_dir,
        pixel_size_um=pixel_size_um,
        image_id=image_id,
        channels_cfg=channels_cfg,
        celltype_cfg=celltype_cfg,
        labels=labels,
        df_pixels=df_pixels,
        shapes=shapes,
        params=params,
        save_outputs=save_outputs,
        make_figures=make_figures,
        native_threads=native_threads,
        support_workers=support_workers,
    )


def rank_celltype_assignment_parameter_sweep_results(
    df_results: pd.DataFrame,
    defined_celltype_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    ranked = df_results.copy()
    if "error" in ranked.columns:
        ranked = ranked[ranked["error"].fillna("") == ""].copy()
    if len(ranked) == 0:
        return ranked

    if defined_celltype_names is None:
        defined_celltype_names = [
            col.replace("count::", "")
            for col in ranked.columns
            if str(col).startswith("count::") and col not in {"count::Unassigned", "count::Ambiguous"}
        ]
    defined_celltype_names = list(defined_celltype_names)

    if "assigned_defined_total" not in ranked.columns:
        total = np.zeros(len(ranked), dtype=float)
        for name in defined_celltype_names:
            col = f"count::{name}"
            if col in ranked.columns:
                total = total + pd.to_numeric(ranked[col], errors="coerce").fillna(0).to_numpy()
        ranked["assigned_defined_total"] = total

    sort_cols = ["assigned_defined_total"]
    ascending = [False]
    if "count::Ambiguous" in ranked.columns:
        sort_cols.append("count::Ambiguous")
        ascending.append(True)
    if "count::Unassigned" in ranked.columns:
        sort_cols.append("count::Unassigned")
        ascending.append(True)
    if "combo_index" in ranked.columns:
        sort_cols.append("combo_index")
        ascending.append(True)
    return ranked.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def recommend_celltype_assignment_parameter_sweep_result(
    df_results: pd.DataFrame,
    defined_celltype_names: Sequence[str] | None = None,
) -> pd.Series | None:
    ranked = rank_celltype_assignment_parameter_sweep_results(df_results, defined_celltype_names=defined_celltype_names)
    if len(ranked) == 0:
        return None
    return ranked.iloc[0]


def run_celltype_assignment_parameter_sweep(
    folder: Path,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    image_id: str,
    channels_cfg: Sequence[Dict[str, Any]],
    celltype_cfg: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    df_pixels: pd.DataFrame,
    shapes: Dict[Tuple[str, str], Tuple[int, int]],
    base_params: CelltypeAssignmentParams,
    sweep_values: Dict[str, Sequence[Any]],
    save_outputs: bool = True,
    parallel_workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ordered_values: Dict[str, List[Any]] = {}
    n_combinations = 1
    for field in CELLTYPE_PARAM_ORDER:
        values = list(sweep_values.get(field, [getattr(base_params, field)]))
        if field == "thresh_mode":
            norm_values = [str(v) for v in values if str(v).strip()]
            if not norm_values:
                norm_values = [str(getattr(base_params, field))]
        elif field in {"min_pos_object_size_px", "min_pos_pix"}:
            norm_values = [max(0, int(v)) for v in values]
        else:
            norm_values = [float(v) for v in values]
        seen = set()
        unique_values = []
        for v in norm_values:
            key = str(v) if field == "thresh_mode" else float(v)
            if key in seen:
                continue
            seen.add(key)
            unique_values.append(v)
        ordered_values[field] = unique_values
        n_combinations *= len(unique_values)

    defined_count_columns = [f"count::{ct['name']}" for ct in celltype_cfg]
    extra_count_columns = ["count::Unassigned", "count::Ambiguous"]
    count_columns = defined_count_columns + extra_count_columns

    combos = list(product(*[ordered_values[field] for field in CELLTYPE_PARAM_ORDER]))

    def evaluate_combo(combo_index: int, combo_values: Tuple[Any, ...]) -> Dict[str, Any]:
        overrides = {field: value for field, value in zip(CELLTYPE_PARAM_ORDER, combo_values)}
        params = replace(base_params, **overrides)
        row: Dict[str, Any] = {
            "combo_index": int(combo_index),
        }
        row.update({CELLTYPE_PARAM_LABELS[field]: getattr(params, field) for field in CELLTYPE_PARAM_ORDER})
        try:
            result = _run_celltype_assignment_impl(
                folder=folder,
                save_dir=save_dir,
                pixel_size_um=pixel_size_um,
                image_id=image_id,
                channels_cfg=channels_cfg,
                celltype_cfg=celltype_cfg,
                labels=labels,
                df_pixels=df_pixels,
                shapes=shapes,
                params=params,
                save_outputs=False,
                make_figures=False,
                native_threads=1,
                support_workers=1,
            )
            counts_map = result["counts"].set_index("celltype")["count"].to_dict() if len(result["counts"]) > 0 else {}
            row["n_cells"] = int(len(result["df_cells"]))
            for col in count_columns:
                ct_name = col.replace("count::", "")
                row[col] = int(counts_map.get(ct_name, 0))
            row["assigned_defined_total"] = int(sum(int(counts_map.get(ct["name"], 0)) for ct in celltype_cfg))
            row["error"] = ""
        except Exception as exc:
            row["n_cells"] = np.nan
            for col in count_columns:
                row[col] = np.nan
            row["assigned_defined_total"] = np.nan
            row["error"] = str(exc)
        return row

    # Run sequentially for stability. The progress callback is invoked after each
    # tested combination so the Streamlit UI can update a progress bar.
    records: List[Dict[str, Any]] = []
    total = len(combos)
    if progress_callback is not None:
        progress_callback(0, total)
    for done, (combo_index, combo_values) in enumerate(enumerate(combos, start=1), start=1):
        records.append(evaluate_combo(combo_index, combo_values))
        if progress_callback is not None:
            progress_callback(done, total)

    df_results = pd.DataFrame(records)
    csv_path = save_dir / "celltype_assignment_parameter_sweep_results.csv"
    json_path = save_dir / "celltype_assignment_parameter_sweep_grid.json"
    fig = make_celltype_assignment_parameter_sweep_figure(df_results, count_columns, celltype_cfg=celltype_cfg)
    svg_path = save_dir / "celltype_assignment_parameter_sweep.svg"
    png_path = save_dir / "celltype_assignment_parameter_sweep.png"

    if save_outputs:
        df_results.to_csv(csv_path, index=False)
        write_json(
            json_path,
            {
                "n_combinations": int(n_combinations),
                "base_params": base_params.to_dict(),
                "candidate_values": {
                    CELLTYPE_PARAM_LABELS[k]: [str(v) if k == "thresh_mode" else float(v) if k not in {"min_pos_object_size_px", "min_pos_pix"} else int(v) for v in vals]
                    for k, vals in ordered_values.items()
                },
                "count_columns": count_columns,
                "execution_mode": "sequential",
            },
        )
        fig.savefig(svg_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

    return {
        "results": df_results,
        "figure": fig,
        "n_combinations": int(n_combinations),
        "candidate_values": ordered_values,
        "count_columns": count_columns,
        "saved_paths": {
            "csv": csv_path,
            "json": json_path,
            "svg": svg_path,
            "png": png_path,
        },
    }
