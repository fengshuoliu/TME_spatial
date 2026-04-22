from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from .io import save_uint16_tiff, valid_pixel_size, write_json
from .visualization import add_scalebar_20um, axis_off


EXCLUDED_CELLTYPES_DEFAULT = ('Unassigned', 'Ambiguous')


def _coerce_grid_size_px(grid_size_um: float, pixel_size_um: Tuple[float, float]) -> tuple[int, int]:
    px_um_x, px_um_y = float(pixel_size_um[0]), float(pixel_size_um[1])
    tile_w_px = max(1, int(round(float(grid_size_um) / max(1e-12, px_um_x))))
    tile_h_px = max(1, int(round(float(grid_size_um) / max(1e-12, px_um_y))))
    return tile_w_px, tile_h_px


def _cluster_label_from_types(celltypes: Sequence[str]) -> str:
    uniq = [str(ct).strip() for ct in celltypes if str(ct).strip()]
    uniq = sorted(dict.fromkeys(uniq))
    return ' + '.join(uniq)


def run_neighborhood_analysis(
    df_cells: pd.DataFrame,
    image_shape: tuple[int, int],
    pixel_size_um: tuple[float, float],
    grid_size_um: float = 20.0,
    exclude_celltypes: Sequence[str] = EXCLUDED_CELLTYPES_DEFAULT,
) -> Dict[str, Any]:
    if not valid_pixel_size(pixel_size_um):
        raise RuntimeError('PIXEL_SIZE_UM missing/invalid — neighborhood analysis needs valid pixel size.')

    required_cols = {'label', 'celltype', 'centroid_x_px', 'centroid_y_px'}
    if not required_cols.issubset(df_cells.columns):
        raise RuntimeError(f'df_cells must contain columns: {required_cols}')

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise RuntimeError(f'Invalid image shape for neighborhood analysis: {image_shape}')

    grid_size_um = float(grid_size_um)
    if grid_size_um <= 0:
        raise RuntimeError('Neighborhood square size (µm) must be > 0.')

    tile_w_px, tile_h_px = _coerce_grid_size_px(grid_size_um, pixel_size_um)
    n_tiles_x = int(np.ceil(width / float(tile_w_px)))
    n_tiles_y = int(np.ceil(height / float(tile_h_px)))

    exclude_set = {str(v) for v in exclude_celltypes}
    valid_cells = df_cells[~df_cells['celltype'].astype(str).isin(exclude_set)].copy()

    cluster_mask = np.zeros((height, width), dtype=np.uint16)

    if len(valid_cells) == 0:
        empty_summary = pd.DataFrame(columns=['cluster_id', 'cluster_label', 'cluster_key', 'n_tiles', 'n_cells', 'tile_fraction'])
        empty_tiles = pd.DataFrame(columns=[
            'tile_row', 'tile_col', 'tile_index', 'x0_px', 'x1_px', 'y0_px', 'y1_px',
            'n_cells', 'celltypes', 'cluster_key', 'cluster_label', 'cluster_id'
        ])
        return {
            'cluster_mask': cluster_mask,
            'cluster_summary': empty_summary,
            'tile_assignments': empty_tiles,
            'grid_size_um': grid_size_um,
            'tile_width_px': tile_w_px,
            'tile_height_px': tile_h_px,
            'n_tiles_x': n_tiles_x,
            'n_tiles_y': n_tiles_y,
            'cluster_labels': [],
            'cluster_keys': [],
            'excluded_celltypes': list(exclude_set),
        }

    valid_cells['tile_col'] = np.clip((valid_cells['centroid_x_px'].to_numpy(float) // tile_w_px).astype(int), 0, n_tiles_x - 1)
    valid_cells['tile_row'] = np.clip((valid_cells['centroid_y_px'].to_numpy(float) // tile_h_px).astype(int), 0, n_tiles_y - 1)

    tile_rows: List[Dict[str, Any]] = []
    for (tile_row, tile_col), group in valid_cells.groupby(['tile_row', 'tile_col'], sort=True):
        celltypes = sorted(dict.fromkeys(group['celltype'].astype(str).tolist()))
        cluster_key = '|'.join(celltypes)
        cluster_label = _cluster_label_from_types(celltypes)
        x0 = int(tile_col * tile_w_px)
        y0 = int(tile_row * tile_h_px)
        x1 = int(min(width, x0 + tile_w_px))
        y1 = int(min(height, y0 + tile_h_px))
        tile_rows.append(
            {
                'tile_row': int(tile_row),
                'tile_col': int(tile_col),
                'tile_index': int(tile_row * n_tiles_x + tile_col),
                'x0_px': x0,
                'x1_px': x1,
                'y0_px': y0,
                'y1_px': y1,
                'n_cells': int(len(group)),
                'celltypes': cluster_label,
                'cluster_key': cluster_key,
                'cluster_label': cluster_label,
            }
        )

    tile_assignments = pd.DataFrame(tile_rows)
    if len(tile_assignments) == 0:
        raise RuntimeError('Neighborhood analysis found no valid occupied tiles after excluding Unassigned/Ambiguous.')

    cluster_order_df = (
        tile_assignments[['cluster_key', 'cluster_label']]
        .drop_duplicates()
        .assign(n_types=lambda df: df['cluster_key'].apply(lambda x: 0 if not str(x) else len(str(x).split('|'))))
        .sort_values(['n_types', 'cluster_label'], ascending=[True, True])
        .reset_index(drop=True)
    )
    cluster_order_df['cluster_id'] = np.arange(1, len(cluster_order_df) + 1, dtype=int)

    cluster_key_to_id = dict(zip(cluster_order_df['cluster_key'], cluster_order_df['cluster_id']))
    tile_assignments['cluster_id'] = tile_assignments['cluster_key'].map(cluster_key_to_id).astype(int)

    for row in tile_assignments.itertuples(index=False):
        cluster_mask[int(row.y0_px):int(row.y1_px), int(row.x0_px):int(row.x1_px)] = int(row.cluster_id)

    cluster_summary = (
        tile_assignments.groupby(['cluster_id', 'cluster_label', 'cluster_key'], as_index=False)
        .agg(n_tiles=('tile_index', 'count'), n_cells=('n_cells', 'sum'))
        .sort_values('cluster_id')
        .reset_index(drop=True)
    )
    total_tiles = float(n_tiles_x * n_tiles_y)
    cluster_summary['tile_fraction'] = cluster_summary['n_tiles'].to_numpy(float) / max(1.0, total_tiles)

    cluster_labels = cluster_summary['cluster_label'].astype(str).tolist()
    cluster_keys = cluster_summary['cluster_key'].astype(str).tolist()

    return {
        'cluster_mask': cluster_mask.astype(np.uint16),
        'cluster_summary': cluster_summary,
        'tile_assignments': tile_assignments.sort_values(['tile_row', 'tile_col']).reset_index(drop=True),
        'grid_size_um': float(grid_size_um),
        'tile_width_px': int(tile_w_px),
        'tile_height_px': int(tile_h_px),
        'n_tiles_x': int(n_tiles_x),
        'n_tiles_y': int(n_tiles_y),
        'cluster_labels': cluster_labels,
        'cluster_keys': cluster_keys,
        'excluded_celltypes': list(exclude_set),
    }



def _filter_neighborhood_display(
    result: Dict[str, Any],
    display_cluster_labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    cluster_mask_raw = result.get('cluster_mask', np.zeros((0, 0), dtype=np.uint16))
    cluster_mask = np.asarray(cluster_mask_raw).astype(np.uint16)

    cluster_summary_raw = result.get('cluster_summary')
    if isinstance(cluster_summary_raw, pd.DataFrame):
        cluster_summary = cluster_summary_raw.copy()
    else:
        cluster_summary = pd.DataFrame()

    tile_assignments_raw = result.get('tile_assignments')
    if isinstance(tile_assignments_raw, pd.DataFrame):
        tile_assignments = tile_assignments_raw.copy()
    else:
        tile_assignments = pd.DataFrame()

    summary_cols = ['cluster_id', 'cluster_label', 'cluster_key', 'n_tiles', 'n_cells', 'tile_fraction']
    for col in summary_cols:
        if col not in cluster_summary.columns:
            cluster_summary[col] = pd.Series(dtype='object' if col in {'cluster_label', 'cluster_key'} else 'float64')
    cluster_summary = cluster_summary[summary_cols].copy()

    tile_cols = [
        'tile_row', 'tile_col', 'tile_index', 'x0_px', 'x1_px', 'y0_px', 'y1_px',
        'n_cells', 'celltypes', 'cluster_key', 'cluster_label', 'cluster_id'
    ]
    for col in tile_cols:
        if col not in tile_assignments.columns:
            tile_assignments[col] = pd.Series(dtype='object' if col in {'celltypes', 'cluster_key', 'cluster_label'} else 'float64')
    tile_assignments = tile_assignments[tile_cols].copy()

    if cluster_mask.size == 0 or len(cluster_summary) == 0:
        return (
            np.zeros_like(cluster_mask, dtype=np.uint16),
            cluster_summary.iloc[0:0].copy(),
            tile_assignments.iloc[0:0].copy(),
            [],
        )

    cluster_summary = cluster_summary.sort_values('cluster_id').reset_index(drop=True)
    available_labels = cluster_summary['cluster_label'].astype(str).tolist()
    available_set = set(available_labels)

    if display_cluster_labels is None:
        selected_labels = list(available_labels)
    else:
        selected_labels = [str(label) for label in display_cluster_labels if str(label) in available_set]
        if not selected_labels:
            selected_labels = list(available_labels)

    filtered_summary = cluster_summary[cluster_summary['cluster_label'].astype(str).isin(selected_labels)].copy().reset_index(drop=True)
    if len(filtered_summary) == 0:
        return (
            np.zeros_like(cluster_mask, dtype=np.uint16),
            filtered_summary,
            tile_assignments.iloc[0:0].copy(),
            [],
        )

    old_ids = filtered_summary['cluster_id'].astype(int).tolist()
    old_to_new = {old_id: idx + 1 for idx, old_id in enumerate(old_ids)}

    filtered_mask = np.zeros_like(cluster_mask, dtype=np.uint16)
    for old_id, new_id in old_to_new.items():
        filtered_mask[cluster_mask == int(old_id)] = np.uint16(new_id)

    filtered_summary['cluster_id'] = np.arange(1, len(filtered_summary) + 1, dtype=int)

    filtered_tiles = tile_assignments[tile_assignments['cluster_label'].astype(str).isin(selected_labels)].copy()
    if len(filtered_tiles) > 0:
        label_to_new_id = dict(zip(filtered_summary['cluster_label'].astype(str), filtered_summary['cluster_id'].astype(int)))
        filtered_tiles['cluster_id'] = filtered_tiles['cluster_label'].astype(str).map(label_to_new_id).astype(int)
        filtered_tiles = filtered_tiles.sort_values(['tile_row', 'tile_col']).reset_index(drop=True)

    return filtered_mask, filtered_summary, filtered_tiles, selected_labels


def make_neighborhood_figure(
    cluster_mask: np.ndarray,
    cluster_summary: pd.DataFrame,
    pixel_size_um: tuple[float, float],
    cluster_colors: Dict[str, str] | None = None,
    title: str = 'Neighborhood clusters',
    display_cluster_labels: Sequence[str] | None = None,
) -> plt.Figure:
    cluster_colors = cluster_colors or {}
    cluster_summary = cluster_summary.copy()
    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    if cluster_mask is None or cluster_mask.size == 0 or len(cluster_summary) == 0:
        ax.text(0.5, 0.5, 'No neighborhood clusters to display', ha='center', va='center', transform=ax.transAxes)
        axis_off(ax)
        return fig

    display_result = {
        'cluster_mask': np.asarray(cluster_mask).astype(np.uint16),
        'cluster_summary': cluster_summary,
        'tile_assignments': pd.DataFrame(),
    }
    cluster_mask_display, cluster_summary_display, _, _ = _filter_neighborhood_display(
        display_result,
        display_cluster_labels=display_cluster_labels,
    )

    if cluster_mask_display.size == 0 or len(cluster_summary_display) == 0:
        ax.text(0.5, 0.5, 'No neighborhood clusters selected for display', ha='center', va='center', transform=ax.transAxes)
        axis_off(ax)
        return fig

    palette = [(0.0, 0.0, 0.0)]
    legend_handles = []

    for row in cluster_summary_display.itertuples(index=False):
        color_hex = str(cluster_colors.get(str(row.cluster_label), '#cccccc'))
        palette.append(mcolors.to_rgb(color_hex))
        legend_handles.append(mpatches.Patch(color=color_hex, label=str(row.cluster_label)))

    cmap = ListedColormap(palette)
    ax.imshow(cluster_mask_display, cmap=cmap, origin='upper', interpolation='nearest', vmin=0, vmax=len(cluster_summary_display))
    axis_off(ax)
    add_scalebar_20um(ax, cluster_mask_display.shape, float(pixel_size_um[0]), bar_um=20.0, color='white', lw=4, pad_frac=0.05)
    ax.text(
        0.985,
        0.985,
        title,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=13,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=(0, 0, 0, 0.28), edgecolor='none'),
    )

    if legend_handles:
        ncol = 1 if len(legend_handles) <= 8 else 2
        ax.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fontsize=9,
            ncol=ncol,
            borderaxespad=0.0,
            title='Clusters',
            title_fontsize=10,
        )
        fig.subplots_adjust(right=0.76)
    else:
        fig.subplots_adjust(right=0.98)

    return fig


def save_neighborhood_analysis_outputs(
    result: Dict[str, Any],
    save_dir: Path,
    pixel_size_um: tuple[float, float],
    cluster_colors: Dict[str, str],
    display_cluster_labels: Sequence[str] | None = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cluster_mask, cluster_summary, tile_assignments, selected_labels = _filter_neighborhood_display(
        result,
        display_cluster_labels=display_cluster_labels,
    )

    figure = make_neighborhood_figure(
        cluster_mask=cluster_mask,
        cluster_summary=cluster_summary,
        pixel_size_um=pixel_size_um,
        cluster_colors=cluster_colors,
        title=f"Neighborhood clusters ({float(result.get('grid_size_um', 20.0)):.1f} µm grid)",
    )

    mask_tiff = save_dir / 'neighborhood_cluster_mask_uint16.tiff'
    summary_csv = save_dir / 'neighborhood_cluster_summary.csv'
    tiles_csv = save_dir / 'neighborhood_tile_assignments.csv'
    params_json = save_dir / 'neighborhood_params.json'
    svg_path = save_dir / 'neighborhood_clusters.svg'
    png_path = save_dir / 'neighborhood_clusters.png'
    tiff_path = save_dir / 'neighborhood_clusters.tiff'

    if save_outputs:
        save_uint16_tiff(mask_tiff, cluster_mask)
        cluster_summary.to_csv(summary_csv, index=False)
        tile_assignments.to_csv(tiles_csv, index=False)
        write_json(
            params_json,
            {
                'grid_size_um': float(result.get('grid_size_um', 20.0)),
                'tile_width_px': int(result.get('tile_width_px', 0)),
                'tile_height_px': int(result.get('tile_height_px', 0)),
                'n_tiles_x': int(result.get('n_tiles_x', 0)),
                'n_tiles_y': int(result.get('n_tiles_y', 0)),
                'excluded_celltypes': list(result.get('excluded_celltypes', [])),
                'display_cluster_labels': list(selected_labels),
                'cluster_colors': {str(k): str(v) for k, v in cluster_colors.items()},
            },
        )
        figure.savefig(svg_path, dpi=600, bbox_inches='tight', pad_inches=0)
        figure.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
        figure.savefig(tiff_path, dpi=600, bbox_inches='tight', pad_inches=0)

    return {
        'figure': figure,
        'display_cluster_labels': list(selected_labels),
        'saved_paths': {
            'mask_tiff': mask_tiff,
            'summary_csv': summary_csv,
            'tiles_csv': tiles_csv,
            'params_json': params_json,
            'svg': svg_path,
            'png': png_path,
            'tiff': tiff_path,
        },
    }
