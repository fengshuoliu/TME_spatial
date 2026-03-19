from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    save_outputs: bool = True,
) -> Dict[str, Any]:
    if not celltype_cfg:
        raise RuntimeError("CELLTYPE_CFG is empty.")

    folder = Path(folder)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not valid_pixel_size(pixel_size_um):
        raise RuntimeError("PIXEL_SIZE_UM missing/invalid. Please provide valid x/y pixel sizes.")

    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    px_area_um2 = px_um_x * px_um_y

    cpu = os.cpu_count() or 1
    target_threads = int(os.environ.get("OMP_NUM_THREADS", max(1, cpu - 1)))
    target_threads = max(1, target_threads)
    numba_threads = min(target_threads, 24)

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
    ch2hex = {c["channel"]: c.get("color_hex", "#ffffff") for c in channels_cfg}
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

    r_voronoi_um = 3.0
    r_buffer_um = 2.0
    r_vote_um = 3.0
    tophat_r_um = 1.0
    gauss_sigma_um = 0.5
    thresh_mode = "global_otsu"
    min_pos_pix = 5

    r_voronoi_px = um_to_px_iso(r_voronoi_um)
    r_buffer_px = um_to_px_iso(r_buffer_um)
    r_vote_px = um_to_px_iso(r_vote_um)
    tophat_px = um_to_px_iso(tophat_r_um)
    gauss_sigma = max(0.1, gauss_sigma_um / max(1e-12, np.sqrt(px_um_x * px_um_y)))

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
        from numba import get_num_threads, njit, prange, set_num_threads

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
        pos = morphology.remove_small_objects(pos, min_size=9)
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
        df_cells[f"{marker_key}_pos"] = df_cells.get(f"{marker_key}_pos_pix", 0) >= min_pos_pix

    def is_pos(row: pd.Series, marker_key: str) -> bool:
        if marker_key == NUC_KEY:
            return bool(row.get("NUCLEUS_pos", False))
        col = f"{marker_key}_pos"
        return bool(row.get(col, False))

    def match_simple(ct: Dict[str, Any], row: pd.Series) -> bool:
        all_pos = [marker_name_to_key(marker) for marker in ct.get("all_pos", [])]
        all_neg = [marker_name_to_key(marker) for marker in ct.get("all_neg", [])]
        any_groups = [
            [marker_name_to_key(marker) for marker in group] for group in ct.get("any_pos_groups", [])
        ]

        if not all(is_pos(row, mk) for mk in all_pos):
            return False
        if not all((not is_pos(row, mk)) for mk in all_neg):
            return False
        for group in any_groups:
            if group and (not any(is_pos(row, mk) for mk in group)):
                return False
        return True

    compiled_expr: List[Any] = []
    for ct in celltype_cfg:
        if ct.get("mode") == "expr":
            expr = normalize_expr(ct.get("expr", ""))
            try:
                compiled_expr.append(compile(expr, "<celltype_expr>", "eval") if expr else None)
            except Exception:
                compiled_expr.append(None)
        else:
            compiled_expr.append(None)

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

    k_types = len(celltype_cfg)
    celltype_id = np.zeros(len(df_cells), dtype=np.uint16)
    celltype_name = np.array([""] * len(df_cells), dtype=object)

    for i in range(len(df_cells)):
        row = df_cells.iloc[i]
        assigned = False
        for k, ct in enumerate(celltype_cfg, start=1):
            if ct.get("mode") == "simple":
                ok = match_simple(ct, row)
            else:
                ok = match_expr(k - 1, row)
            if ok:
                celltype_id[i] = k
                celltype_name[i] = ct["name"]
                assigned = True
                break
        if not assigned:
            celltype_id[i] = 1
            celltype_name[i] = celltype_cfg[0]["name"]

    df_cells["celltype_id"] = celltype_id.astype(int)
    df_cells["celltype"] = celltype_name

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

    slices_list = ndi.find_objects(labels)
    margin = int(max(r_voronoi_px, r_buffer_px) + 2)
    assign_map_list = list(assign_maps.values())

    def expand_slice(slice_pair, margin_local: int, h_local: int, w_local: int):
        sy, sx = slice_pair
        y0 = max(0, sy.start - margin_local)
        y1 = min(h_local, sy.stop + margin_local)
        x0 = max(0, sx.start - margin_local)
        x1 = min(w_local, sx.stop + margin_local)
        return slice(y0, y1), slice(x0, x1)

    def support_for_label(label_id: int):
        slc0 = slices_list[label_id - 1]
        if slc0 is None:
            return label_id, None, None
        slc = expand_slice(slc0, margin, h, w)
        lbl_loc = labels[slc]
        own_loc = owner_map[slc]
        dist_loc = dist_outside[slc]
        ct = int(celltype_id_by_label[label_id])

        support = lbl_loc == label_id
        for amap in assign_map_list:
            support |= amap[slc] == label_id

        if ct >= 1 and ct <= k_types and (not type_has_marker[ct]):
            support |= (own_loc == label_id) & (dist_loc <= r_buffer_px)

        support = ndi.binary_fill_holes(support)
        return label_id, ct, (slc, support)

    results = Parallel(n_jobs=target_threads, prefer="threads", batch_size=64)(
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

    if save_outputs:
        save_uint16_tiff(save_dir / "celltypes_mask_uint16.tiff", celltype_mask.astype(np.uint16))
        df_cells.to_csv(save_dir / "cells_summary.csv", index=False)
        counts.to_csv(save_dir / "celltype_counts.csv", index=False)
        thresholds_df.to_csv(save_dir / "marker_assignment_thresholds.csv", index=False)
        save_celltype_config(celltype_cfg, save_dir)

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

    panel_svg = save_dir / "celltypes_panel.svg"
    panel_tiff = save_dir / "celltypes_panel.tiff"

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

    if save_outputs:
        panel_fig.savefig(panel_svg, dpi=600, bbox_inches="tight", pad_inches=0)
        panel_fig.savefig(panel_tiff, dpi=600, bbox_inches="tight", pad_inches=0)

    split_svg = save_dir / "celltypes_split_panels.svg"
    split_tiff = save_dir / "celltypes_split_panels.tiff"

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
        split_fig.savefig(split_svg, dpi=600, bbox_inches="tight", pad_inches=0)
        split_fig.savefig(split_tiff, dpi=600, bbox_inches="tight", pad_inches=0)

    return {
        "df_cells": df_cells,
        "counts": counts,
        "celltype_mask": celltype_mask,
        "thresholds": thresholds_df,
        "nuc_channel": nuc_channel,
        "panel_figure": panel_fig,
        "split_figure": split_fig,
        "saved_paths": {
            "celltype_mask_tiff": save_dir / "celltypes_mask_uint16.tiff",
            "cells_summary_csv": save_dir / "cells_summary.csv",
            "celltype_counts_csv": save_dir / "celltype_counts.csv",
            "marker_assignment_thresholds_csv": save_dir / "marker_assignment_thresholds.csv",
            "panel_svg": panel_svg,
            "panel_tiff": panel_tiff,
            "split_svg": split_svg,
            "split_tiff": split_tiff,
            "celltype_config_json": save_dir / "celltype_config.json",
        },
    }
