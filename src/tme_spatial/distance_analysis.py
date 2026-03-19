from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import ttest_rel
from skimage import segmentation

from .io import load_any_tiff, safe_name, valid_pixel_size


def run_nearest_neighbor_analysis(
    df_cells: pd.DataFrame,
    celltype_cfg: Sequence[Dict[str, Any]],
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    target_type: str,
    query_types: Sequence[str],
    save_outputs: bool = True,
) -> Dict[str, Any]:
    if not valid_pixel_size(pixel_size_um):
        raise RuntimeError("PIXEL_SIZE_UM is missing/invalid — needed to convert distances to µm.")

    required_cols = {"label", "celltype", "centroid_x_px", "centroid_y_px"}
    if not required_cols.issubset(df_cells.columns):
        raise RuntimeError(f"df_cells must contain columns: {required_cols}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])

    def nearest_pairs(df: pd.DataFrame, target_type_local: str, query_type_local: str) -> pd.DataFrame:
        tdf = df[df["celltype"].astype(str) == str(target_type_local)].copy()
        qdf = df[df["celltype"].astype(str) == str(query_type_local)].copy()

        if len(tdf) == 0:
            raise ValueError(f"No cells found for target_type={target_type_local!r}")
        if len(qdf) == 0:
            raise ValueError(f"No cells found for query_type={query_type_local!r}")

        tx = tdf["centroid_x_px"].to_numpy(float) * px_um_x
        ty = tdf["centroid_y_px"].to_numpy(float) * px_um_y
        qx = qdf["centroid_x_px"].to_numpy(float) * px_um_x
        qy = qdf["centroid_y_px"].to_numpy(float) * px_um_y

        t_xy = np.column_stack([tx, ty])
        q_xy = np.column_stack([qx, qy])

        tree = cKDTree(q_xy)
        dist, idx = tree.query(t_xy, k=1, workers=-1)

        q_match = qdf.iloc[idx].reset_index(drop=True)
        out = pd.DataFrame(
            {
                "target_type": target_type_local,
                "query_type": query_type_local,
                "target_label": tdf["label"].to_numpy(int),
                "query_label": q_match["label"].to_numpy(int),
                "target_x_um": tx,
                "target_y_um": ty,
                "query_x_um": q_match["centroid_x_px"].to_numpy(float) * px_um_x,
                "query_y_um": q_match["centroid_y_px"].to_numpy(float) * px_um_y,
                "dist_um": dist.astype(float),
            }
        )
        out["dx_um"] = out["query_x_um"] - out["target_x_um"]
        out["dy_um"] = out["query_y_um"] - out["target_y_um"]
        return out

    def paired_ttests(df_long: pd.DataFrame, qtypes: Sequence[str]):
        wide = df_long.pivot(index="target_label", columns="query_type", values="dist_um").reindex(columns=qtypes)
        comparisons: List[Tuple[int, int]] = []
        if len(qtypes) == 2:
            comparisons = [(0, 1)]
        elif len(qtypes) > 2:
            comparisons = [(0, i) for i in range(1, len(qtypes))]

        rows: List[Dict[str, Any]] = []
        for a, b in comparisons:
            qa, qb = qtypes[a], qtypes[b]
            x = wide[qa].to_numpy(float)
            y = wide[qb].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            n_pairs = int(mask.sum())
            if n_pairs < 2:
                t_val = np.nan
                p_val = np.nan
            else:
                t_val, p_val = ttest_rel(x[mask], y[mask], nan_policy="omit")
            rows.append({"ref": qa, "cmp": qb, "n_pairs": n_pairs, "t": t_val, "p": p_val})
        return pd.DataFrame(rows), wide

    def format_p(p_val: float) -> str:
        if not np.isfinite(p_val):
            return "p=NA"
        if p_val < 1e-4:
            return f"p={p_val:.1e}"
        return f"p={p_val:.4f}"

    def add_bracket(ax, x1: float, x2: float, y: float, h: float, text: str, fontsize: int = 11):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
        ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=fontsize)

    def plot_box_scatter(df_long: pd.DataFrame, title: str):
        qtypes = list(dict.fromkeys(df_long["query_type"].tolist()))
        data = [df_long.loc[df_long["query_type"] == q, "dist_um"].to_numpy(float) for q in qtypes]

        fig, ax = plt.subplots(figsize=(1.7 * max(3, len(qtypes)), 5))
        ax.boxplot(data, labels=qtypes, showfliers=False)

        ttest_df, wide = paired_ttests(df_long, qtypes)

        x_positions = np.arange(1, len(qtypes) + 1, dtype=float)
        for _, row in wide.iterrows():
            y_vals = row.to_numpy(float)
            mask = np.isfinite(y_vals)
            if mask.sum() >= 2:
                ax.plot(x_positions[mask], y_vals[mask], linewidth=0.8, alpha=0.25)

        rng = np.random.default_rng(0)
        for i, q in enumerate(qtypes, start=1):
            y_vals = df_long.loc[df_long["query_type"] == q, "dist_um"].to_numpy(float)
            x_vals = i + rng.uniform(-0.12, 0.12, size=len(y_vals))
            ax.scatter(x_vals, y_vals, s=18, alpha=0.8)

        ax.set_ylabel("Nearest distance (µm)")
        ax.set_xlabel("Query cell type")
        ax.set_title(title)
        plt.xticks(rotation=30, ha="right")

        if len(qtypes) >= 2:
            all_y = np.concatenate([d[np.isfinite(d)] for d in data if len(d) > 0]) if data else np.array([0.0])
            y_max = float(np.nanmax(all_y)) if all_y.size else 1.0
            y_min = float(np.nanmin(all_y)) if all_y.size else 0.0
            span = max(1e-6, y_max - y_min)
            base_y = y_max + 0.06 * span
            step_h = 0.08 * span
            bracket_h = 0.02 * span

            if len(qtypes) == 2:
                pairs = [(1, 2)]
            else:
                pairs = [(1, i + 1) for i in range(1, len(qtypes))]

            for k, (x1, x2) in enumerate(pairs):
                row = ttest_df.iloc[k] if k < len(ttest_df) else None
                text = format_p(row["p"]) if row is not None else "p=NA"
                y_val = base_y + k * step_h
                add_bracket(ax, x1, x2, y_val, bracket_h, text)

            ax.set_ylim(top=base_y + (len(pairs) + 1) * step_h)

        plt.tight_layout()
        return fig, ttest_df

    df = df_cells.copy()
    df["celltype"] = df["celltype"].astype(str)

    all_rows: List[pd.DataFrame] = []
    for query_type in query_types:
        if query_type != target_type:
            all_rows.append(nearest_pairs(df, target_type, query_type))
        else:
            tdf = df[df["celltype"] == target_type].copy()
            if len(tdf) < 2:
                raise ValueError(f"Need at least 2 cells of type {target_type!r} for self-type nearest neighbor.")
            tx = tdf["centroid_x_px"].to_numpy(float) * px_um_x
            ty = tdf["centroid_y_px"].to_numpy(float) * px_um_y
            t_xy = np.column_stack([tx, ty])
            tree = cKDTree(t_xy)
            dist, idx = tree.query(t_xy, k=2, workers=-1)
            nn_dist = dist[:, 1]
            nn_idx = idx[:, 1]
            q_match = tdf.iloc[nn_idx].reset_index(drop=True)

            out_self = pd.DataFrame(
                {
                    "target_type": target_type,
                    "query_type": target_type,
                    "target_label": tdf["label"].to_numpy(int),
                    "query_label": q_match["label"].to_numpy(int),
                    "target_x_um": tx,
                    "target_y_um": ty,
                    "query_x_um": q_match["centroid_x_px"].to_numpy(float) * px_um_x,
                    "query_y_um": q_match["centroid_y_px"].to_numpy(float) * px_um_y,
                    "dist_um": nn_dist.astype(float),
                }
            )
            out_self["dx_um"] = out_self["query_x_um"] - out_self["target_x_um"]
            out_self["dy_um"] = out_self["query_y_um"] - out_self["target_y_um"]
            all_rows.append(out_self)

    df_long = pd.concat(all_rows, ignore_index=True)
    df_long["pair_id"] = df_long["target_label"].astype(str)

    base = safe_name(f"nn_{target_type}__to__{'__'.join(query_types)}", "nn")
    csv_path = save_dir / f"nearest_neighbor_distances__{base}.csv"
    svg_path = save_dir / f"nearest_neighbor_distances__{base}.svg"
    png_path = save_dir / f"nearest_neighbor_distances__{base}.png"

    if save_outputs:
        df_long.to_csv(csv_path, index=False)

    figure, ttest_df = plot_box_scatter(df_long, title=f"Nearest distances to target: {target_type}")
    if save_outputs:
        figure.savefig(svg_path, dpi=600, bbox_inches="tight")
        figure.savefig(png_path, dpi=600, bbox_inches="tight")

    return {
        "distances": df_long,
        "ttests": ttest_df,
        "figure": figure,
        "saved_paths": {
            "csv": csv_path,
            "svg": svg_path,
            "png": png_path,
        },
    }


def discover_boundary_masks(save_dir: Path, celltype_cfg: Sequence[Dict[str, Any]], df_cells: pd.DataFrame) -> List[Tuple[str, Path]]:
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


def run_boundary_distance_analysis(
    df_cells: pd.DataFrame,
    celltype_cfg: Sequence[Dict[str, Any]],
    celltype_mask: np.ndarray,
    save_dir: Path,
    pixel_size_um: Tuple[float, float],
    boundary_mask_path: Path,
    boundary_name: str,
    query_types: Sequence[str],
    region_filter: str = "all",
    save_outputs: bool = True,
) -> Dict[str, Any]:
    if not valid_pixel_size(pixel_size_um):
        raise RuntimeError("PIXEL_SIZE_UM missing/invalid — needed to compute distances in µm.")

    required_cols = {"label", "celltype", "centroid_x_px", "centroid_y_px"}
    if not required_cols.issubset(df_cells.columns):
        raise RuntimeError(f"df_cells must contain columns: {required_cols}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    height, width = celltype_mask.shape

    if not Path(boundary_mask_path).exists():
        raise FileNotFoundError(f"Boundary mask file not found: {boundary_mask_path}")

    region_mask = load_any_tiff(Path(boundary_mask_path)) > 0
    if region_mask.shape != (height, width):
        raise RuntimeError(
            f"Boundary mask shape {region_mask.shape} does not match current celltype_mask shape {(height, width)}."
        )

    boundary_mask = segmentation.find_boundaries(region_mask.astype(bool), mode="thick").astype(bool)
    if not np.any(boundary_mask):
        raise RuntimeError("Boundary mask is empty (no boundary pixels).")

    dist_map_um = ndi.distance_transform_edt(~boundary_mask, sampling=(px_um_y, px_um_x))

    rows: List[pd.DataFrame] = []
    for query_type in query_types:
        sub = df_cells[df_cells["celltype"].astype(str) == str(query_type)].copy()
        if len(sub) == 0:
            continue

        cy = np.clip(np.rint(sub["centroid_y_px"].to_numpy(float)).astype(int), 0, height - 1)
        cx = np.clip(np.rint(sub["centroid_x_px"].to_numpy(float)).astype(int), 0, width - 1)

        inside = region_mask[cy, cx]
        if region_filter == "inside":
            keep = inside
        elif region_filter == "outside":
            keep = ~inside
        else:
            keep = np.ones_like(inside, dtype=bool)

        if not np.any(keep):
            continue

        cyk, cxk = cy[keep], cx[keep]
        subk = sub.iloc[np.where(keep)[0]].copy()
        dist_um = dist_map_um[cyk, cxk].astype(float)

        out_df = pd.DataFrame(
            {
                "boundary_mask_file": Path(boundary_mask_path).name,
                "boundary_name": boundary_name,
                "query_celltype": query_type,
                "query_label": subk["label"].to_numpy(int),
                "centroid_x_px": subk["centroid_x_px"].to_numpy(float),
                "centroid_y_px": subk["centroid_y_px"].to_numpy(float),
                "centroid_x_um": subk["centroid_x_px"].to_numpy(float) * px_um_x,
                "centroid_y_um": subk["centroid_y_px"].to_numpy(float) * px_um_y,
                "inside_region": region_mask[cyk, cxk].astype(bool),
                "dist_to_boundary_um": dist_um,
            }
        )
        rows.append(out_df)

    if not rows:
        raise RuntimeError("No cells found after applying the selected boundary filter.")

    df_long = pd.concat(rows, ignore_index=True)

    def plot_box_scatter(df_long_local: pd.DataFrame, title: str):
        qtypes = list(dict.fromkeys(df_long_local["query_celltype"].tolist()))
        data = [df_long_local.loc[df_long_local["query_celltype"] == q, "dist_to_boundary_um"].to_numpy(float) for q in qtypes]

        fig, ax = plt.subplots(figsize=(1.7 * max(3, len(qtypes)), 5))
        ax.boxplot(data, labels=qtypes, showfliers=False)

        rng = np.random.default_rng(0)
        for i, q in enumerate(qtypes, start=1):
            y_vals = df_long_local.loc[df_long_local["query_celltype"] == q, "dist_to_boundary_um"].to_numpy(float)
            x_vals = i + rng.uniform(-0.12, 0.12, size=len(y_vals))
            ax.scatter(x_vals, y_vals, s=18, alpha=0.8)

        ax.set_ylabel("Shortest distance to boundary (µm)")
        ax.set_xlabel("Query cell type")
        ax.set_title(title)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig

    base = safe_name(
        f"to_boundary__{Path(boundary_mask_path).stem}__from__{'__'.join(query_types)}__{region_filter}",
        "boundary",
    )
    csv_path = save_dir / f"dist_to_boundary__{base}.csv"
    svg_path = save_dir / f"dist_to_boundary__{base}.svg"
    png_path = save_dir / f"dist_to_boundary__{base}.png"

    if save_outputs:
        df_long.to_csv(csv_path, index=False)

    figure = plot_box_scatter(df_long, title=f"Distance to boundary: {Path(boundary_mask_path).stem}")
    if save_outputs:
        figure.savefig(svg_path, dpi=600, bbox_inches="tight")
        figure.savefig(png_path, dpi=600, bbox_inches="tight")

    return {
        "distances": df_long,
        "figure": figure,
        "saved_paths": {
            "csv": csv_path,
            "svg": svg_path,
            "png": png_path,
        },
    }
