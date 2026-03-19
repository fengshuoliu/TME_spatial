from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation
from skimage.measure import find_contours

from .io import save_uint16_tiff, write_json
from .models import NucleiParams
from .visualization import norm_clip
from .io import to_image


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


def run_nuclei_segmentation(
    df_pixels: pd.DataFrame,
    shapes: Dict[Tuple[str, str], Tuple[int, int]],
    image_id: str,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    params: NucleiParams,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Port of the notebook's nuclei segmentation logic with minimal changes.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    px_area_um2 = px_um_x * px_um_y

    def um_to_px_x(value_um: float) -> int:
        return max(1, int(round(value_um / px_um_x)))

    def um_to_px_y(value_um: float) -> int:
        return max(1, int(round(value_um / px_um_y)))

    def um_to_px_iso(value_um: float) -> int:
        return max(1, int(round(value_um / np.sqrt(px_um_x * px_um_y))))

    min_area_px = int(np.pi * (um_to_px_iso(params.min_diam_um) / 2.0) ** 2 * 0.25)
    max_area_px = int(np.pi * (um_to_px_iso(params.max_diam_um) / 2.0) ** 2 * 4.0)
    tophat_r_px = um_to_px_iso(params.tophat_radius_um)
    gauss_sigma = max(0.1, params.gauss_sigma_um / np.sqrt(px_um_x * px_um_y))
    local_win = um_to_px_iso(params.local_win_um) | 1
    h_maxima_val = max(0.25, um_to_px_iso(params.h_maxima_um) * 1.0)
    seed_min_dist = um_to_px_iso(params.seed_min_dist_um)

    dapi = to_image(df_pixels, shapes, image_id, params.nucleus_channel).astype(np.float32)

    low, high = np.nanpercentile(dapi, [1, 99.8])
    dapi_norm = np.clip((dapi - low) / max(1e-6, (high - low)), 0, 1)

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

    grad = filters.sobel(dapi_smooth)
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

    rng = np.random.default_rng(42)
    palette = np.vstack([[0, 0, 0], rng.random((max(n_nuclei, 1), 3))])
    cmap = ListedColormap(palette)

    h, w = labels.shape
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

    panel_svg = save_dir / "nuclei_segmentation_panel.svg"
    panel_tiff = save_dir / "nuclei_segmentation_panel.tiff"
    if save_outputs:
        fig.savefig(panel_svg, dpi=300, bbox_inches="tight")
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
            "panel_tiff": panel_tiff,
            "params_json": params_json,
        },
    }
