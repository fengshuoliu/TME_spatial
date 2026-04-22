from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .io import to_image, valid_pixel_size


def rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def generate_distinct_hex(n: int = 72, s: float = 0.78, v: float = 0.95) -> List[str]:
    import colorsys

    cols: List[str] = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / float(n), s, v)
        cols.append(rgb_to_hex(r, g, b))
    return cols


COMMON_FIRST = ["#dc0000", "#00ff00", "#0008e5", "#ffffff", "#ff00ff", "#00ffff", "#ffff00"]
COLOR_HEX_LIST = list(dict.fromkeys(COMMON_FIRST + generate_distinct_hex(n=84, s=0.78, v=0.95)))


def norm_clip(arr: np.ndarray, lo_percentile: float = 0.0, hi_percentile: float = 99.8) -> np.ndarray:
    a = arr.astype(float, copy=False)
    lo_v = np.nanpercentile(a, lo_percentile)
    hi_v = np.nanpercentile(a, hi_percentile)
    if hi_v <= lo_v:
        return np.zeros_like(a, dtype=float)
    a = (a - lo_v) / (hi_v - lo_v)
    return np.clip(a, 0, 1)


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(hex_color), dtype=float)


def axis_off(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_scalebar_20um(
    ax: plt.Axes,
    arr_shape: Sequence[int],
    px_um_x: float | None,
    bar_um: float = 20.0,
    color: str = "white",
    lw: float = 4.0,
    pad_frac: float = 0.05,
) -> None:
    h = int(arr_shape[0])
    w = int(arr_shape[1])
    if px_um_x is None or float(px_um_x) <= 0:
        return
    bar_px = int(round(float(bar_um) / float(px_um_x)))
    if bar_px <= 0:
        return
    pad_x = int(round(pad_frac * w))
    pad_y = int(round(pad_frac * h))
    x_end = w - pad_x
    x_start = max(pad_x, x_end - bar_px)
    y = h - pad_y
    ax.plot([x_start, x_end], [y, y], color=color, lw=lw, solid_capstyle="butt")


def add_channel_labels(
    ax: plt.Axes,
    names: Sequence[str],
    hex_colors: Sequence[str],
    x: float = 0.985,
    y0: float = 0.985,
) -> None:
    n = max(1, len(names))
    dy = min(0.06, 0.90 / n)
    fontsize = 14 if n <= 6 else max(8, int(14 * 6 / n))
    y = y0
    for name, color_hex in zip(names, hex_colors):
        ax.text(
            x,
            y,
            name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color=color_hex,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.25), edgecolor="none"),
        )
        y -= dy


def add_colored_type_text(
    ax: plt.Axes,
    names: Sequence[str],
    colors_hex: Sequence[str],
    title: str | None = None,
    x: float = 0.985,
    y0: float = 0.985,
    fontsize: int = 14,
) -> None:
    n = max(1, len(names))
    dy = min(0.06, 0.90 / n)
    fs = fontsize if n <= 6 else max(8, int(fontsize * 6 / n))
    y = y0
    if title:
        ax.text(
            x,
            y,
            title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fs,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.30), edgecolor="none"),
        )
        y -= dy
    for name, color_hex in zip(names, colors_hex):
        ax.text(
            x,
            y,
            name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fs,
            fontweight="bold",
            color=color_hex,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.25), edgecolor="none"),
        )
        y -= dy


def build_celltype_cmap(celltype_cfg: Sequence[Dict[str, Any]]) -> ListedColormap:
    ct_hex = [ct["color_hex"] for ct in celltype_cfg]
    ct_rgb = [(0, 0, 0)] + [mcolors.to_rgb(h) for h in ct_hex]
    return ListedColormap(ct_rgb)


def save_figure_svg_and_png(fig, save_path: Path | None = None, png_dpi: int = 300) -> None:
    if not save_path:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".png":
        png_path = save_path
        svg_path = save_path.with_suffix(".svg")
    else:
        svg_path = save_path.with_suffix(".svg")
        png_path = save_path.with_suffix(".png")
    fig.savefig(str(svg_path), bbox_inches="tight", pad_inches=0)
    fig.savefig(str(png_path), dpi=png_dpi, bbox_inches="tight", pad_inches=0)


def overlay_multi_channels(
    df,
    shapes,
    image_id: str,
    channels_cfg: Sequence[Dict[str, Any]],
    overlay_channels: Sequence[str],
    white_channel: str | None = None,
    white_weight: float = 0.0,
    clip_hi: float = 99.8,
    pixel_size_um: Tuple[float, float] | None = None,
    save_path: Path | None = None,
):
    if overlay_channels is None or len(overlay_channels) == 0:
        raise ValueError("overlay_channels is empty.")

    ch2hex = {c["channel"]: c["color_hex"] for c in channels_cfg}

    rgb = None
    h = w = None

    for channel in overlay_channels:
        if channel not in ch2hex:
            raise KeyError(f"Channel {channel!r} not found in channels_cfg.")
        img = norm_clip(to_image(df, shapes, image_id, channel), hi_percentile=clip_hi)
        if rgb is None:
            h, w = img.shape
            rgb = np.zeros((h, w, 3), dtype=float)
        layer = np.clip(img[..., None] * hex_to_rgb01(ch2hex[channel]), 0.0, 1.0)
        rgb = 1.0 - (1.0 - rgb) * (1.0 - layer)

    if white_channel and white_channel != "None" and float(white_weight) > 0:
        wimg = norm_clip(to_image(df, shapes, image_id, white_channel), hi_percentile=clip_hi)
        wlayer = np.clip(float(white_weight) * wimg, 0.0, 1.0)[..., None]
        rgb = 1.0 - (1.0 - rgb) * (1.0 - wlayer)

    rgb = np.clip(rgb, 0.0, 1.0)

    px_um_x = pixel_size_um[0] if valid_pixel_size(pixel_size_um) else None

    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.imshow(rgb, origin="upper")
    axis_off(ax)
    add_channel_labels(ax, overlay_channels, [ch2hex[ch] for ch in overlay_channels])
    add_scalebar_20um(ax, (h, w), px_um_x, bar_um=20.0, color="white", lw=4)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save_path:
        save_figure_svg_and_png(fig, save_path=Path(save_path), png_dpi=300)

    return fig, rgb


def plot_split_channels(
    df,
    shapes,
    image_id: str,
    channels_cfg: Sequence[Dict[str, Any]],
    pixel_size_um: Tuple[float, float] | None = None,
    clip_hi: float = 99.8,
    save_path: Path | None = None,
):
    channel_names = [c["channel"] for c in channels_cfg]
    n = len(channel_names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
    axes = np.array(axes).ravel()

    px_um_x = pixel_size_um[0] if valid_pixel_size(pixel_size_um) else None

    for ax, channel_cfg in zip(axes, channels_cfg):
        channel = channel_cfg["channel"]
        color_hex = channel_cfg["color_hex"]
        img = norm_clip(to_image(df, shapes, image_id, channel), hi_percentile=clip_hi)
        rgb = np.clip(img[..., None] * hex_to_rgb01(color_hex), 0.0, 1.0)

        ax.imshow(rgb, origin="upper")
        axis_off(ax)
        add_channel_labels(ax, [channel], [color_hex])
        add_scalebar_20um(ax, img.shape, px_um_x, bar_um=20.0, color="white", lw=4)

    for ax in axes[len(channel_names):]:
        ax.set_visible(False)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)

    if save_path:
        save_figure_svg_and_png(fig, save_path=Path(save_path), png_dpi=300)

    return fig


def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    return buffer.getvalue()
