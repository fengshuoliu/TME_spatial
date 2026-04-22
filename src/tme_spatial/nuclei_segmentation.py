from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - optional runtime dependency guard
    Parallel = None
    delayed = None

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional runtime dependency guard
    threadpool_limits = None

from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation
from skimage.measure import find_contours

from .io import save_uint16_tiff, to_image, write_json
from .models import NucleiParams
from .visualization import norm_clip

SWEEP_PARAM_ORDER: List[str] = [
    "min_diam_um",
    "max_diam_um",
    "tophat_radius_um",
    "gauss_sigma_um",
    "local_win_um",
    "local_offset",
    "h_maxima_um",
    "seed_min_dist_um",
    "watershed_compactness",
    "post_resplit_mult",
]

SWEEP_PARAM_LABELS: Dict[str, str] = {
    "min_diam_um": "MIN_DIAM_UM",
    "max_diam_um": "MAX_DIAM_UM",
    "tophat_radius_um": "TOPHAT_RADIUS_UM",
    "gauss_sigma_um": "GAUSS_SIGMA_UM",
    "local_win_um": "LOCAL_WIN_UM",
    "local_offset": "LOCAL_OFFSET",
    "h_maxima_um": "H_MAXIMA_UM",
    "seed_min_dist_um": "SEED_MIN_DIST_UM",
    "watershed_compactness": "WATERSHED_COMPACTNESS",
    "post_resplit_mult": "POST_RESPLIT_MULT",
}


DEFAULT_CPU_COUNT = os.cpu_count() or 1


CANONICAL_NUCLEI_PARAM_FIELDS: List[str] = ["nucleus_channel", *SWEEP_PARAM_ORDER]


def _canonical_nuclei_param_key(key: Any) -> str | None:
    text = str(key).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in CANONICAL_NUCLEI_PARAM_FIELDS:
        return lowered
    if lowered == "nucleus_channel":
        return "nucleus_channel"
    for field, label in SWEEP_PARAM_LABELS.items():
        if lowered == label.lower():
            return field
    alias_map = {
        "nucleuschannel": "nucleus_channel",
        "nucleus-channel": "nucleus_channel",
    }
    return alias_map.get(lowered)


def _coerce_nuclei_params(params_like: Any, overrides: Dict[str, Any] | None = None) -> NucleiParams:
    raw: Dict[str, Any] = {}
    if isinstance(params_like, dict):
        raw.update(params_like)
    elif hasattr(params_like, "to_dict") and callable(getattr(params_like, "to_dict")):
        try:
            raw.update(params_like.to_dict())
        except Exception:
            pass
    for field in CANONICAL_NUCLEI_PARAM_FIELDS:
        if hasattr(params_like, field):
            raw[field] = getattr(params_like, field)

    if overrides:
        raw.update(overrides)

    clean: Dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _canonical_nuclei_param_key(key)
        if canonical is None:
            continue
        clean[canonical] = value

    if "nucleus_channel" not in clean or clean["nucleus_channel"] in (None, ""):
        raise RuntimeError("Nucleus channel is missing from nuclei segmentation parameters.")

    for field in SWEEP_PARAM_ORDER:
        if field in clean:
            clean[field] = float(clean[field])

    return NucleiParams(**{field: clean[field] for field in CANONICAL_NUCLEI_PARAM_FIELDS if field in clean})


def _thread_limit_context(n_threads: int | None):
    if threadpool_limits is None or n_threads is None:
        return nullcontext()
    try:
        n_threads = int(n_threads)
    except Exception:
        return nullcontext()
    if n_threads < 1:
        return nullcontext()
    return threadpool_limits(limits=n_threads)


def _iter_combo_chunks(
    ordered_values: Dict[str, List[float]],
    chunk_size: int,
) -> Iterable[List[Tuple[int, Tuple[float, ...]]]]:
    chunk: List[Tuple[int, Tuple[float, ...]]] = []
    for combo_index, combo in enumerate(product(*[ordered_values[field] for field in SWEEP_PARAM_ORDER]), start=1):
        chunk.append((combo_index, tuple(float(v) for v in combo)))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _evaluate_sweep_combo_chunk(
    combo_chunk: Sequence[Tuple[int, Tuple[float, ...]]],
    base_params: NucleiParams,
    dapi: np.ndarray,
    dapi_norm: np.ndarray,
    pixel_size_um: Tuple[float, float],
    native_threads: int | None = 1,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with _thread_limit_context(native_threads):
        for combo_index, combo in combo_chunk:
            overrides = {field: float(value) for field, value in zip(SWEEP_PARAM_ORDER, combo)}
            params = _coerce_nuclei_params(base_params, overrides)
            row: Dict[str, Any] = {
                "combo_index": int(combo_index),
                "nucleus_channel": params.nucleus_channel,
            }
            row.update({SWEEP_PARAM_LABELS[field]: float(getattr(params, field)) for field in SWEEP_PARAM_ORDER})
            try:
                labels = segment_nuclei_from_prepared_images(dapi, dapi_norm, pixel_size_um, params)
                n_nuclei = int(labels.max())
                positive_px = int((labels > 0).sum())
                row["n_nuclei"] = n_nuclei
                row["positive_pixel_fraction"] = float(positive_px / labels.size) if labels.size > 0 else 0.0
                row["mean_pixels_per_nucleus"] = float(positive_px / n_nuclei) if n_nuclei > 0 else 0.0
                row["error"] = ""
            except Exception as exc:  # pragma: no cover - defensive UI guard
                row["n_nuclei"] = np.nan
                row["positive_pixel_fraction"] = np.nan
                row["mean_pixels_per_nucleus"] = np.nan
                row["error"] = str(exc)
            rows.append(row)
    return rows



def pick_nucleus_channel(channels: Sequence[str]) -> str | None:
    if not channels:
        return None
    preferred_exact = ["DAPI", "HOECHST", "HOECHST33342", "H33342", "H342", "NUCLEUS", "NUCLEAR"]
    upper = {channel.upper(): channel for channel in channels}
    for preferred in preferred_exact:
        if preferred in upper:
            return upper[preferred]
    preferred_sub = ["DAPI", "HOECHST", "H333", "NUC"]
    for channel in channels:
        upper_channel = channel.upper()
        if any(token in upper_channel for token in preferred_sub):
            return channel
    return channels[0]


def load_nucleus_channel_image(
    df_pixels: pd.DataFrame,
    shapes: Dict[Tuple[str, str], Tuple[int, int]],
    image_id: str,
    nucleus_channel: str,
) -> np.ndarray:
    return to_image(df_pixels, shapes, image_id, nucleus_channel).astype(np.float32)


def normalize_nucleus_image(dapi: np.ndarray) -> np.ndarray:
    low, high = np.nanpercentile(dapi, [1, 99.8])
    return np.clip((dapi - low) / max(1e-6, (high - low)), 0, 1)


def _pixel_converters(pixel_size_um: Tuple[float, float]):
    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])

    def um_to_px_x(value_um: float) -> int:
        return max(1, int(round(value_um / px_um_x)))

    def um_to_px_y(value_um: float) -> int:
        return max(1, int(round(value_um / px_um_y)))

    def um_to_px_iso(value_um: float) -> int:
        return max(1, int(round(value_um / np.sqrt(px_um_x * px_um_y))))

    return px_um_x, px_um_y, um_to_px_x, um_to_px_y, um_to_px_iso


def segment_nuclei_from_prepared_images(
    dapi: np.ndarray,
    dapi_norm: np.ndarray,
    pixel_size_um: Tuple[float, float],
    params: NucleiParams,
) -> np.ndarray:
    px_um_x, px_um_y, _, _, um_to_px_iso = _pixel_converters(pixel_size_um)

    min_area_px = int(np.pi * (um_to_px_iso(params.min_diam_um) / 2.0) ** 2 * 0.25)
    max_area_px = int(np.pi * (um_to_px_iso(params.max_diam_um) / 2.0) ** 2 * 4.0)
    tophat_r_px = um_to_px_iso(params.tophat_radius_um)
    iso_scale_um = np.sqrt(px_um_x * px_um_y)
    gauss_sigma = max(0.0, params.gauss_sigma_um / max(1e-12, iso_scale_um))
    local_win = um_to_px_iso(params.local_win_um) | 1
    h_maxima_val = max(0.0, float(params.h_maxima_um) / max(1e-12, iso_scale_um))
    seed_min_dist = max(0, int(round(float(params.seed_min_dist_um) / max(1e-12, iso_scale_um))))

    selem = morphology.disk(tophat_r_px)
    dapi_th = morphology.white_tophat(dapi_norm, footprint=selem)
    dapi_smooth = filters.gaussian(dapi_th, sigma=gauss_sigma, preserve_range=True)

    local_thr = filters.threshold_local(dapi_smooth, block_size=local_win, offset=params.local_offset)
    bw = dapi_smooth > local_thr

    bw = morphology.remove_small_objects(bw, min_size=max(32, min_area_px))
    bw = morphology.remove_small_holes(bw, area_threshold=max(64, min_area_px // 2))

    lbl_tmp = measure.label(bw, connectivity=2)
    if max_area_px > 0:
        props_tmp = measure.regionprops(lbl_tmp)
        drop_ids = {prop.label for prop in props_tmp if prop.area > max_area_px}
        if drop_ids:
            bw = ~np.isin(lbl_tmp, list(drop_ids))

    dist = ndi.distance_transform_edt(bw)
    dist_s = filters.gaussian(dist, sigma=0.5, preserve_range=True)

    coords = feature.peak_local_max(
        dist_s,
        labels=bw,
        min_distance=max(1, seed_min_dist),
        threshold_abs=h_maxima_val,
        exclude_border=False,
    )

    markers = np.zeros_like(dist_s, dtype=np.int32)
    if len(coords) > 0:
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    else:
        markers = measure.label(bw)

    grad = filters.sobel(dapi_th if gauss_sigma <= 0 else dapi_smooth)
    labels = segmentation.watershed(
        grad,
        markers=markers,
        mask=bw,
        compactness=params.watershed_compactness,
        watershed_line=True,
    )

    props1 = measure.regionprops(labels)
    if len(props1) >= 10:
        areas = np.array([prop.area for prop in props1], dtype=float)
        med_area = np.median(areas[areas > 0]) if np.any(areas > 0) else 0
        if med_area > 0:
            big_ids = [prop.label for prop in props1 if prop.area >= params.post_resplit_mult * med_area]
            if big_ids:
                lab_refined = labels.copy()
                for lab_id in big_ids:
                    region_mask = labels == lab_id
                    if region_mask.sum() < 20:
                        continue

                    bbox = measure.regionprops(region_mask.astype(np.uint8))[0].bbox
                    minr, minc, maxr, maxc = bbox
                    pad = 4
                    minr = max(0, minr - pad)
                    minc = max(0, minc - pad)
                    maxr = min(labels.shape[0], maxr + pad)
                    maxc = min(labels.shape[1], maxc + pad)

                    sub_mask = region_mask[minr:maxr, minc:maxc]
                    if sub_mask.sum() < 20:
                        continue

                    sub_dist = ndi.distance_transform_edt(sub_mask)
                    sub_dist_s = filters.gaussian(sub_dist, sigma=0.5, preserve_range=True)

                    sub_coords = feature.peak_local_max(
                        sub_dist_s,
                        labels=sub_mask,
                        min_distance=max(1, seed_min_dist),
                        threshold_abs=h_maxima_val,
                        exclude_border=False,
                    )
                    if len(sub_coords) <= 1:
                        continue

                    sub_mark = np.zeros_like(sub_dist_s, dtype=np.int32)
                    sub_mark[tuple(sub_coords.T)] = np.arange(1, len(sub_coords) + 1)

                    sub_grad = grad[minr:maxr, minc:maxc]
                    sub_lab = segmentation.watershed(
                        sub_grad,
                        markers=sub_mark,
                        mask=sub_mask,
                        compactness=params.watershed_compactness,
                        watershed_line=True,
                    )

                    max_cur = lab_refined.max()
                    sub_lab_w = sub_lab.copy()
                    sub_lab_w[sub_lab_w > 0] += max_cur

                    roi = lab_refined[minr:maxr, minc:maxc]
                    roi[sub_mask] = sub_lab_w[sub_mask]
                    lab_refined[minr:maxr, minc:maxc] = roi

                labels = lab_refined

    labels = measure.label(labels > 0, connectivity=2) * (labels > 0)
    return labels.astype(np.int32)


def summarize_nuclei_labels(
    labels: np.ndarray,
    dapi: np.ndarray,
    pixel_size_um: Tuple[float, float],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    px_um_x, px_um_y, _, _, _ = _pixel_converters(pixel_size_um)
    px_area_um2 = px_um_x * px_um_y
    n_nuclei = int(labels.max())

    props = measure.regionprops_table(
        labels,
        intensity_image=dapi,
        properties=(
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "centroid",
            "bbox",
            "mean_intensity",
            "max_intensity",
        ),
    )
    df_props = pd.DataFrame(props)
    if len(df_props) > 0:
        df_props.rename(
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
        df_props["centroid_x_um"] = df_props["centroid_x_px"] * px_um_x
        df_props["centroid_y_um"] = df_props["centroid_y_px"] * px_um_y
        df_props["area_um2"] = df_props["area"] * px_area_um2
        df_props["perimeter_um"] = df_props["perimeter"] * np.sqrt(px_um_x * px_um_y)

    boundaries: List[Dict[str, Any]] = []
    for label_id in range(1, n_nuclei + 1):
        mask_i = labels == label_id
        contours = find_contours(mask_i.astype(float), 0.5)
        if not contours:
            continue
        contour = max(contours, key=lambda arr: arr.shape[0])
        xy_px = np.column_stack([contour[:, 1], contour[:, 0]])
        xy_um = xy_px * np.array([px_um_x, px_um_y])
        boundaries.append(
            {
                "label": int(label_id),
                "boundary_px": xy_px.tolist(),
                "boundary_um": xy_um.tolist(),
            }
        )

    return df_props, boundaries


def make_nuclei_segmentation_figure(
    dapi: np.ndarray,
    labels_u16: np.ndarray,
    image_id: str,
    pixel_size_um: Tuple[float, float],
    params: NucleiParams,
) -> plt.Figure:
    px_um_x, px_um_y, _, _, _ = _pixel_converters(pixel_size_um)
    n_nuclei = int(labels_u16.max())

    rng = np.random.default_rng(42)
    palette = np.vstack([[0, 0, 0], rng.random((max(n_nuclei, 1), 3))])
    cmap = ListedColormap(palette)

    h, w = labels_u16.shape
    extent = [0, w * px_um_x, h * px_um_y, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(norm_clip(dapi, hi_percentile=99.8), cmap="gray", origin="upper", extent=extent)
    axes[0].set_title(f"{image_id} — {params.nucleus_channel}")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")

    axes[1].imshow(labels_u16, cmap=cmap, origin="upper", extent=extent, interpolation="nearest")
    axes[1].set_title("Nuclei (random colors)")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")
    return fig


def run_nuclei_segmentation(
    df_pixels: pd.DataFrame,
    shapes: Dict[Tuple[str, str], Tuple[int, int]],
    image_id: str,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    params: NucleiParams,
    save_outputs: bool = True,
    native_threads: int | None = None,
) -> Dict[str, Any]:
    """
    Port of the notebook's nuclei segmentation logic with minimal changes.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    params = _coerce_nuclei_params(params)

    dapi = load_nucleus_channel_image(df_pixels, shapes, image_id, params.nucleus_channel)
    dapi_norm = normalize_nucleus_image(dapi)
    with _thread_limit_context(native_threads):
        labels = segment_nuclei_from_prepared_images(dapi, dapi_norm, pixel_size_um, params)
        n_nuclei = int(labels.max())
        df_props, boundaries = summarize_nuclei_labels(labels, dapi, pixel_size_um)
    labels_u16 = labels.astype(np.uint16)

    summary_csv = save_dir / "nuclei_summary.csv"
    boundaries_json = save_dir / "nuclei_boundaries.json"
    tiff_path = save_dir / "nuclei_labels_uint16.tiff"
    params_json = save_dir / "nuclei_params.json"

    if save_outputs:
        df_props.to_csv(summary_csv, index=False)
        boundaries_json.write_text(json.dumps(boundaries))
        save_uint16_tiff(tiff_path, labels_u16)
        write_json(params_json, params.to_dict())

    fig = make_nuclei_segmentation_figure(dapi, labels_u16, image_id, pixel_size_um, params)

    panel_svg = save_dir / "nuclei_segmentation_panel.svg"
    panel_png = save_dir / "nuclei_segmentation_panel.png"
    panel_tiff = save_dir / "nuclei_segmentation_panel.tiff"
    if save_outputs:
        fig.savefig(panel_svg, bbox_inches="tight")
        fig.savefig(panel_png, dpi=300, bbox_inches="tight")
        fig.savefig(panel_tiff, dpi=300, bbox_inches="tight")

    return {
        "labels": labels,
        "labels_u16": labels_u16,
        "n_nuclei": n_nuclei,
        "df_props": df_props,
        "boundaries": boundaries,
        "params": params.to_dict(),
        "figure": fig,
        "saved_paths": {
            "summary_csv": summary_csv,
            "boundaries_json": boundaries_json,
            "labels_tiff": tiff_path,
            "panel_svg": panel_svg,
            "panel_png": panel_png,
            "panel_tiff": panel_tiff,
            "params_json": params_json,
        },
    }


def make_nuclei_parameter_sweep_figure(df_results: pd.DataFrame) -> plt.Figure:
    plot_df = df_results.copy()
    if "combo_index" in plot_df.columns:
        plot_df = plot_df.sort_values("combo_index")

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    if len(plot_df) > 0:
        ax.plot(plot_df["combo_index"], plot_df["n_nuclei"], marker="o", linewidth=1)
        best_idx = plot_df["n_nuclei"].astype(float).idxmax()
        best_row = plot_df.loc[best_idx]
        ax.annotate(
            f"best #{int(best_row['combo_index'])}: {int(best_row['n_nuclei'])} nuclei",
            xy=(best_row["combo_index"], best_row["n_nuclei"]),
            xytext=(10, 10),
            textcoords="offset points",
        )
    ax.set_xlabel("Parameter combination index")
    ax.set_ylabel("Segmented nuclei count")
    ax.set_title("Nuclei parameter sweep")
    ax.grid(alpha=0.3)
    return fig



def rank_nuclei_parameter_sweep_results(df_results: pd.DataFrame) -> pd.DataFrame:
    ranked = df_results.copy()
    if "error" in ranked.columns:
        ranked = ranked[ranked["error"].fillna("") == ""].copy()
    if len(ranked) == 0:
        return ranked
    sort_cols = ["n_nuclei", "positive_pixel_fraction", "mean_pixels_per_nucleus", "combo_index"]
    ascending = [False, False, False, True]
    available_cols = [col for col in sort_cols if col in ranked.columns]
    ascending = ascending[: len(available_cols)]
    return ranked.sort_values(available_cols, ascending=ascending).reset_index(drop=True)


def recommend_nuclei_parameter_sweep_result(df_results: pd.DataFrame) -> pd.Series | None:
    ranked = rank_nuclei_parameter_sweep_results(df_results)
    if len(ranked) == 0:
        return None
    return ranked.iloc[0]


def run_nuclei_parameter_sweep(
    df_pixels: pd.DataFrame,
    shapes: Dict[Tuple[str, str], Tuple[int, int]],
    image_id: str,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    base_params: NucleiParams,
    sweep_values: Dict[str, Sequence[float]],
    save_outputs: bool = True,
    max_combinations: int | None = None,
    parallel_workers: int = 1,
    parallel_backend: str = "loky",
    native_threads_per_worker: int | None = 1,
) -> Dict[str, Any]:
    """
    Run a nuclei-parameter sweep.

    Notes
    -----
    - `max_combinations` is kept only for backward compatibility with older app.py files.
      It is no longer enforced as a hard cap.
    - `parallel_workers` controls how many parameter combinations are evaluated concurrently.
    - `native_threads_per_worker` caps the BLAS/OpenMP threads used inside each worker to
      avoid oversubscription during parallel scans.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_params = _coerce_nuclei_params(base_params)

    ordered_values: Dict[str, List[float]] = {}
    n_combinations = 1
    for field in SWEEP_PARAM_ORDER:
        values = sweep_values.get(field, [getattr(base_params, field)])
        values = [float(v) for v in values]
        if not values:
            values = [float(getattr(base_params, field))]
        ordered_values[field] = values
        n_combinations *= len(values)

    dapi = load_nucleus_channel_image(df_pixels, shapes, image_id, base_params.nucleus_channel)
    dapi_norm = normalize_nucleus_image(dapi)

    safe_backend = str(parallel_backend or "loky").strip().lower()
    if safe_backend not in {"loky", "threading"}:
        safe_backend = "loky"

    try:
        parallel_workers = int(parallel_workers)
    except Exception:
        parallel_workers = 1
    parallel_workers = max(1, min(parallel_workers, int(n_combinations)))

    if native_threads_per_worker is not None:
        try:
            native_threads_per_worker = max(1, int(native_threads_per_worker))
        except Exception:
            native_threads_per_worker = 1

    if parallel_workers > 1 and Parallel is not None and delayed is not None:
        chunk_size = max(1, min(128, math.ceil(n_combinations / max(1, parallel_workers * 4))))
        chunk_results = Parallel(
            n_jobs=parallel_workers,
            backend=safe_backend,
            max_nbytes="16M",
            mmap_mode="r",
            batch_size=1,
            verbose=0,
        )(
            delayed(_evaluate_sweep_combo_chunk)(
                combo_chunk,
                base_params,
                dapi,
                dapi_norm,
                pixel_size_um,
                native_threads=native_threads_per_worker,
            )
            for combo_chunk in _iter_combo_chunks(ordered_values, chunk_size)
        )
        records = [row for chunk in chunk_results for row in chunk]
    else:
        records = _evaluate_sweep_combo_chunk(
            list(_iter_combo_chunks(ordered_values, max(1, int(n_combinations))))[0] if n_combinations > 0 else [],
            base_params,
            dapi,
            dapi_norm,
            pixel_size_um,
            native_threads=native_threads_per_worker,
        )

    df_results = pd.DataFrame(records)
    if len(df_results) > 0 and "combo_index" in df_results.columns:
        df_results = df_results.sort_values("combo_index").reset_index(drop=True)

    csv_path = save_dir / "nuclei_parameter_sweep_results.csv"
    json_path = save_dir / "nuclei_parameter_sweep_grid.json"
    fig = make_nuclei_parameter_sweep_figure(df_results[df_results["error"].fillna("") == ""])
    svg_path = save_dir / "nuclei_parameter_sweep.svg"
    png_path = save_dir / "nuclei_parameter_sweep.png"

    if save_outputs:
        df_results.to_csv(csv_path, index=False)
        write_json(
            json_path,
            {
                "nucleus_channel": base_params.nucleus_channel,
                "n_combinations": int(n_combinations),
                "base_params": base_params.to_dict(),
                "candidate_values": {SWEEP_PARAM_LABELS[k]: [float(v) for v in vals] for k, vals in ordered_values.items()},
                "parallel_config": {
                    "parallel_workers": int(parallel_workers),
                    "parallel_backend": safe_backend,
                    "native_threads_per_worker": None if native_threads_per_worker is None else int(native_threads_per_worker),
                    "joblib_available": bool(Parallel is not None and delayed is not None),
                    "threadpoolctl_available": bool(threadpool_limits is not None),
                    "cpu_count": int(DEFAULT_CPU_COUNT),
                },
            },
        )
        fig.savefig(svg_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

    return {
        "results": df_results,
        "figure": fig,
        "n_combinations": int(n_combinations),
        "candidate_values": {field: [float(v) for v in vals] for field, vals in ordered_values.items()},
        "parallel_config": {
            "parallel_workers": int(parallel_workers),
            "parallel_backend": safe_backend,
            "native_threads_per_worker": None if native_threads_per_worker is None else int(native_threads_per_worker),
            "joblib_available": bool(Parallel is not None and delayed is not None),
            "threadpoolctl_available": bool(threadpool_limits is not None),
            "cpu_count": int(DEFAULT_CPU_COUNT),
        },
        "saved_paths": {
            "csv": csv_path,
            "json": json_path,
            "svg": svg_path,
            "png": png_path,
        },
    }
