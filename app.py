\
import os
from pathlib import Path

# Mirror notebook Cell 1 before importing numpy/scipy/skimage/numba.
CPU_COUNT = os.cpu_count() or 4
N_THREADS = max(1, CPU_COUNT - 1)
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("KMP_WARNINGS", "0")

import json
import shutil
import uuid
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.tme_spatial.celltype_assignment import (
    COLOR_HEX_LIST,
    default_celltype,
    guess_nuclear_channel,
    marker_choices_for_ui,
    run_celltype_assignment,
    save_celltype_config,
    token_mapping_for_ui,
)
from src.tme_spatial.distance_analysis import (
    discover_boundary_masks,
    run_boundary_distance_analysis,
    run_nearest_neighbor_analysis,
)
from src.tme_spatial.io import (
    discover_text_image_files,
    files_to_long_df,
    list_output_files,
    pipeline_config_to_json_dict,
    resolve_folder,
    save_uploaded_file_bytes,
    valid_pixel_size,
    write_json,
    zip_directory_bytes,
    load_any_tiff,
)
from src.tme_spatial.models import ChannelConfig, NucleiParams, PipelineConfig, RegionParams
from src.tme_spatial.nuclei_segmentation import pick_nucleus_channel, run_nuclei_segmentation
from src.tme_spatial.region_analysis import discover_boundary_mask_files, run_region_boundary_analysis
from src.tme_spatial.visualization import COMMON_FIRST, overlay_multi_channels, plot_split_channels


st.set_page_config(page_title="TME Spatial", layout="wide")
APP_ROOT = Path.cwd().resolve()
TMP_ROOT = Path("/tmp/tme_spatial_streamlit_sessions")


def session_workspace_root() -> Path:
    return TMP_ROOT / st.session_state["session_id"]


def session_input_dir() -> Path:
    return session_workspace_root() / "inputs"


def session_output_dir() -> Path:
    return session_workspace_root() / "outs"


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
    if "config" not in st.session_state:
        st.session_state["config"] = None
    if "available_files" not in st.session_state:
        st.session_state["available_files"] = []
    if "data_result" not in st.session_state:
        st.session_state["data_result"] = None
    if "nuclei_result" not in st.session_state:
        st.session_state["nuclei_result"] = None
    if "celltype_items" not in st.session_state:
        st.session_state["celltype_items"] = []
    if "celltype_cfg" not in st.session_state:
        st.session_state["celltype_cfg"] = None
    if "assignment_result" not in st.session_state:
        st.session_state["assignment_result"] = None
    if "region_result" not in st.session_state:
        st.session_state["region_result"] = None
    if "nn_result" not in st.session_state:
        st.session_state["nn_result"] = None
    if "boundary_result" not in st.session_state:
        st.session_state["boundary_result"] = None
    if "local_folder_input" not in st.session_state:
        st.session_state["local_folder_input"] = ""
    if "input_mode_radio" not in st.session_state:
        st.session_state["input_mode_radio"] = "Local folder path"
    if "n_channels" not in st.session_state:
        st.session_state["n_channels"] = 3


def clear_state_keys(keys: Sequence[str]) -> None:
    for key in keys:
        st.session_state[key] = None


def invalidate_after_config_change() -> None:
    clear_state_keys(
        [
            "data_result",
            "nuclei_result",
            "assignment_result",
            "region_result",
            "nn_result",
            "boundary_result",
        ]
    )


def invalidate_after_nuclei_change() -> None:
    clear_state_keys(["assignment_result", "region_result", "nn_result", "boundary_result"])


def invalidate_after_celltypes_change() -> None:
    clear_state_keys(["assignment_result", "region_result", "nn_result", "boundary_result"])


def invalidate_after_assignment_change() -> None:
    clear_state_keys(["region_result", "nn_result", "boundary_result"])


def make_celltype_item(index: int) -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:8],
        "default_name": f"celltype_{index + 1}",
        "default_color": COLOR_HEX_LIST[index % len(COLOR_HEX_LIST)],
    }


def ensure_celltype_items() -> None:
    if not st.session_state["celltype_items"]:
        st.session_state["celltype_items"] = [make_celltype_item(0)]


def add_celltype_item() -> None:
    items = list(st.session_state["celltype_items"])
    items.append(make_celltype_item(len(items)))
    st.session_state["celltype_items"] = items


def remove_celltype_item(item_id: str) -> None:
    items = [item for item in st.session_state["celltype_items"] if item["id"] != item_id]
    if not items:
        items = [make_celltype_item(0)]
    st.session_state["celltype_items"] = items


def move_celltype_item(item_id: str, delta: int) -> None:
    items = list(st.session_state["celltype_items"])
    idx = next((i for i, item in enumerate(items) if item["id"] == item_id), None)
    if idx is None:
        return
    new_idx = idx + delta
    if new_idx < 0 or new_idx >= len(items):
        return
    items[idx], items[new_idx] = items[new_idx], items[idx]
    st.session_state["celltype_items"] = items


def reset_session() -> None:
    workspace = session_workspace_root()
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    keys_to_drop = list(st.session_state.keys())
    for key in keys_to_drop:
        del st.session_state[key]
    st.rerun()


def ensure_pixels_loaded() -> None:
    if st.session_state["data_result"] is not None:
        return
    config: PipelineConfig | None = st.session_state["config"]
    if config is None:
        raise RuntimeError("Please save the configuration first.")
    df_pixels, shapes = files_to_long_df(
        folder=config.folder,
        channels_cfg=[channel.to_dict() for channel in config.channels],
        image_id=config.image_id,
        pixel_size_um=config.pixel_size_um if valid_pixel_size(config.pixel_size_um) else None,
        unit="um",
    )
    st.session_state["data_result"] = {
        "df_pixels": df_pixels,
        "shapes": shapes,
    }


def ensure_labels_available():
    if st.session_state["nuclei_result"] is not None:
        return st.session_state["nuclei_result"]["labels"]
    config: PipelineConfig | None = st.session_state["config"]
    if config is None:
        raise RuntimeError("Please save the configuration first.")
    label_path = config.save_dir / "nuclei_labels_uint16.tiff"
    if not label_path.exists():
        raise RuntimeError("No nuclei labels are available yet. Run nuclei segmentation first.")
    return load_any_tiff(label_path).astype("int32")


def ensure_assignment_outputs_available():
    if st.session_state["assignment_result"] is not None:
        return st.session_state["assignment_result"]
    config: PipelineConfig | None = st.session_state["config"]
    if config is None:
        raise RuntimeError("Please save the configuration first.")
    mask_path = config.save_dir / "celltypes_mask_uint16.tiff"
    cells_csv = config.save_dir / "cells_summary.csv"
    if not mask_path.exists() or not cells_csv.exists():
        raise RuntimeError("No cell-type assignment outputs are available yet. Run the assignment step first.")
    celltype_mask = load_any_tiff(mask_path).astype("uint16")
    df_cells = pd.read_csv(cells_csv)
    counts_path = config.save_dir / "celltype_counts.csv"
    counts = pd.read_csv(counts_path) if counts_path.exists() else df_cells["celltype"].value_counts().rename_axis("celltype").reset_index(name="count")
    result = {
        "celltype_mask": celltype_mask,
        "df_cells": df_cells,
        "counts": counts,
    }
    st.session_state["assignment_result"] = result
    return result


def collect_channel_cfg(available_files: Sequence[str]) -> List[ChannelConfig]:
    rows: List[ChannelConfig] = []
    n_channels = int(st.session_state.get("n_channels", 0))
    for idx in range(n_channels):
        file_name = st.session_state.get(f"channel_file_{idx}")
        marker_name = (st.session_state.get(f"channel_marker_{idx}") or "").strip()
        color_hex = st.session_state.get(f"channel_color_{idx}") or COMMON_FIRST[min(idx, len(COMMON_FIRST) - 1)]
        if not file_name:
            raise RuntimeError(f"Channel {idx + 1} is missing a file selection.")
        if file_name not in available_files:
            raise RuntimeError(f"Selected file {file_name!r} is not in the available input file list.")
        if not marker_name:
            marker_name = Path(file_name).stem
        rows.append(ChannelConfig(file=file_name, channel=marker_name, color_hex=color_hex))
    return rows


def current_channel_names_from_widgets() -> List[str]:
    n_channels = int(st.session_state.get("n_channels", 0))
    names: List[str] = []
    for idx in range(n_channels):
        marker_name = (st.session_state.get(f"channel_marker_{idx}") or "").strip()
        file_name = st.session_state.get(f"channel_file_{idx}") or ""
        names.append(marker_name or Path(file_name).stem or f"channel_{idx + 1}")
    return names


def build_and_save_config(available_files: Sequence[str], uploaded_files) -> None:
    if int(st.session_state.get("x_px", 0)) <= 0 or int(st.session_state.get("y_px", 0)) <= 0:
        raise RuntimeError("x (px) and y (px) must both be > 0.")
    pixel_size_um = (
        float(st.session_state.get("x_um", 0.0)) / int(st.session_state.get("x_px", 1)),
        float(st.session_state.get("y_um", 0.0)) / int(st.session_state.get("y_px", 1)),
    )
    channels = collect_channel_cfg(available_files)
    overlay_channels = list(st.session_state.get("overlay_channels", []))
    if not overlay_channels:
        overlay_channels = [channel.channel for channel in channels]
    white_channel = st.session_state.get("white_channel")
    white_weight = float(st.session_state.get("white_weight", 0.0))

    input_mode = st.session_state["input_mode_radio"]
    if input_mode == "Local folder path":
        folder = resolve_folder(st.session_state.get("local_folder_input", ""), APP_ROOT)
        if not folder.exists() or not folder.is_dir():
            raise RuntimeError(f"Folder not found: {folder}")
        save_dir = folder / "outs"
    else:
        if not uploaded_files:
            raise RuntimeError("Please upload at least one CSV/TXT file.")
        workspace_inputs = session_input_dir()
        workspace_outputs = session_output_dir()
        workspace_inputs.mkdir(parents=True, exist_ok=True)
        workspace_outputs.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_files:
            save_uploaded_file_bytes(uploaded.name, uploaded.getvalue(), workspace_inputs)
        folder = workspace_inputs
        save_dir = workspace_outputs

    previous_config: PipelineConfig | None = st.session_state.get("config")
    previous_channel_names = [channel.channel for channel in previous_config.channels] if previous_config else None
    new_channel_names = [channel.channel for channel in channels]

    config = PipelineConfig(
        folder=folder,
        save_dir=save_dir,
        pixel_size_um=pixel_size_um,
        image_id="FieldA",
        channels=channels,
        overlay_channels=overlay_channels,
        white_channel=white_channel if white_channel not in {"", "None"} else None,
        white_weight=white_weight,
        input_mode="local" if input_mode == "Local folder path" else "upload",
    )
    config.save_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.save_dir / "config.json", pipeline_config_to_json_dict(config))
    st.session_state["config"] = config
    invalidate_after_config_change()

    if previous_channel_names != new_channel_names:
        st.session_state["celltype_items"] = []
        st.session_state["celltype_cfg"] = None

    ensure_celltype_items()


def save_celltype_cfg_from_widgets(channel_names: Sequence[str]) -> List[Dict[str, Any]]:
    ensure_celltype_items()
    cfg: List[Dict[str, Any]] = []
    for item in st.session_state["celltype_items"]:
        uid = item["id"]
        name = (st.session_state.get(f"ct_name_{uid}") or "").strip()
        color_hex = st.session_state.get(f"ct_color_{uid}") or item["default_color"]
        mode = st.session_state.get(f"ct_mode_{uid}") or "simple"
        if not name:
            continue
        if mode == "simple":
            all_pos = list(dict.fromkeys(st.session_state.get(f"ct_all_pos_{uid}", [])))
            all_neg = list(dict.fromkeys(st.session_state.get(f"ct_all_neg_{uid}", [])))
            group_count = int(st.session_state.get(f"ct_group_count_{uid}", 0))
            any_groups: List[List[str]] = []
            for group_idx in range(group_count):
                group = list(dict.fromkeys(st.session_state.get(f"ct_group_{uid}_{group_idx}", [])))
                if group:
                    any_groups.append(group)
            cfg.append(
                {
                    "name": name,
                    "color_hex": color_hex,
                    "mode": "simple",
                    "all_pos": all_pos,
                    "all_neg": all_neg,
                    "any_pos_groups": any_groups,
                }
            )
        else:
            expr = (st.session_state.get(f"ct_expr_{uid}") or "").strip()
            cfg.append(
                {
                    "name": name,
                    "color_hex": color_hex,
                    "mode": "expr",
                    "expr": expr,
                }
            )
    if not cfg:
        raise RuntimeError("No valid cell types were defined.")
    st.session_state["celltype_cfg"] = cfg
    config: PipelineConfig | None = st.session_state["config"]
    if config is not None:
        save_celltype_config(cfg, config.save_dir)
    invalidate_after_celltypes_change()
    return cfg


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        st.write(f"CPU count: {CPU_COUNT}")
        st.write(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is not None:
            st.write(f"Input folder: `{config.folder}`")
            st.write(f"Output folder: `{config.save_dir}`")
        else:
            st.write("No configuration saved yet.")
        st.info(
            "For a public hosted Streamlit app, visitors cannot point to their own local machine path. "
            "Use upload mode for hosted deployments; use local-folder mode when you run the app locally."
        )
        if st.button("Reset session", type="secondary"):
            reset_session()


def render_config_tab(tab):
    with tab:
        st.subheader("Input source and configuration")
        input_mode = st.radio(
            "Choose how inputs are provided",
            ["Local folder path", "Upload CSV/TXT files"],
            key="input_mode_radio",
            horizontal=True,
        )

        uploaded_files = None
        available_files: List[str] = []

        if input_mode == "Local folder path":
            st.text_input(
                "Folder path (absolute or relative to the repo root)",
                key="local_folder_input",
                placeholder="Example: data/demo1",
            )
            if st.button("Load folder files", key="load_folder_btn"):
                folder = resolve_folder(st.session_state.get("local_folder_input", ""), APP_ROOT)
                if not folder.exists() or not folder.is_dir():
                    st.error(f"Folder not found: {folder}")
                    st.session_state["available_files"] = []
                else:
                    st.session_state["available_files"] = discover_text_image_files(folder)
            available_files = list(st.session_state.get("available_files", []))
            if available_files:
                st.success(f"Found {len(available_files)} CSV/TXT files.")
                st.code("\n".join(available_files))
        else:
            uploaded_files = st.file_uploader(
                "Upload ImageJ-exported text images (.csv/.txt)",
                type=["csv", "txt"],
                accept_multiple_files=True,
                key="uploaded_files_widget",
            )
            available_files = sorted([uploaded.name for uploaded in uploaded_files]) if uploaded_files else []
            st.session_state["available_files"] = available_files
            if available_files:
                st.success(f"Received {len(available_files)} uploaded files.")
                st.code("\n".join(available_files))

        if not available_files:
            st.warning("Load a local folder or upload files to continue.")
            return

        st.number_input("Number of channels", min_value=1, max_value=20, step=1, key="n_channels")

        if st.button("Reset marker names from filenames", key="reset_marker_names"):
            for idx in range(int(st.session_state["n_channels"])):
                file_name = st.session_state.get(f"channel_file_{idx}") or available_files[min(idx, len(available_files) - 1)]
                st.session_state[f"channel_marker_{idx}"] = Path(file_name).stem

        st.markdown("#### Channel selectors")
        n_channels = int(st.session_state["n_channels"])
        for idx in range(n_channels):
            default_file = available_files[min(idx, len(available_files) - 1)]
            if f"channel_file_{idx}" not in st.session_state:
                st.session_state[f"channel_file_{idx}"] = default_file
            if f"channel_marker_{idx}" not in st.session_state:
                st.session_state[f"channel_marker_{idx}"] = Path(default_file).stem
            if f"channel_color_{idx}" not in st.session_state:
                st.session_state[f"channel_color_{idx}"] = COMMON_FIRST[min(idx, len(COMMON_FIRST) - 1)]

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.selectbox(
                    f"Channel {idx + 1} file",
                    options=available_files,
                    index=available_files.index(st.session_state[f"channel_file_{idx}"]) if st.session_state[f"channel_file_{idx}"] in available_files else 0,
                    key=f"channel_file_{idx}",
                )
            with col2:
                st.text_input(f"Marker {idx + 1}", key=f"channel_marker_{idx}")
            with col3:
                st.color_picker(f"Color {idx + 1}", key=f"channel_color_{idx}")

        st.markdown("#### Pixel size")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.number_input("x (µm)", min_value=0.0, step=0.1, key="x_um")
        with col2:
            st.number_input("x (px)", min_value=0, step=1, key="x_px")
        with col3:
            st.number_input("y (µm)", min_value=0.0, step=0.1, key="y_um")
        with col4:
            st.number_input("y (px)", min_value=0, step=1, key="y_px")

        x_px = int(st.session_state.get("x_px", 0))
        y_px = int(st.session_state.get("y_px", 0))
        if x_px > 0 and y_px > 0:
            pixel_size_um = (
                float(st.session_state.get("x_um", 0.0)) / x_px,
                float(st.session_state.get("y_um", 0.0)) / y_px,
            )
            st.caption(f"Computed PIXEL_SIZE_UM = {pixel_size_um}  (x_um/x_px, y_um/y_px)")
        else:
            st.caption("Enter x (px) and y (px) > 0 to compute PIXEL_SIZE_UM.")

        current_channel_names = current_channel_names_from_widgets()
        if "overlay_channels" not in st.session_state or not st.session_state["overlay_channels"]:
            st.session_state["overlay_channels"] = current_channel_names
        if "white_channel" not in st.session_state:
            st.session_state["white_channel"] = "None"
        if "white_weight" not in st.session_state:
            st.session_state["white_weight"] = 0.0

        st.markdown("#### Overlay options")
        st.multiselect(
            "Overlay channels",
            options=current_channel_names,
            default=[name for name in st.session_state.get("overlay_channels", []) if name in current_channel_names] or current_channel_names,
            key="overlay_channels",
        )
        col1, col2 = st.columns([2, 2])
        with col1:
            st.selectbox("White overlay channel", options=["None"] + current_channel_names, key="white_channel")
        with col2:
            st.slider("White overlay weight", min_value=0.0, max_value=1.0, step=0.05, key="white_weight")

        if st.button("Save configuration", type="primary"):
            try:
                build_and_save_config(available_files, uploaded_files)
                st.success("Configuration saved.")
                config: PipelineConfig = st.session_state["config"]
                st.json(pipeline_config_to_json_dict(config))
            except Exception as exc:
                st.error(str(exc))

        config: PipelineConfig | None = st.session_state.get("config")
        if config is not None:
            st.markdown("#### Current configuration")
            st.json(pipeline_config_to_json_dict(config))


def render_overlay_tab(tab):
    with tab:
        st.subheader("Load inputs and create overlay figures")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is None:
            st.warning("Save the configuration first.")
            return

        if st.button("Load inputs and generate overlay", key="run_overlay_btn"):
            try:
                with st.spinner("Loading CSV/TXT grids and building figures..."):
                    ensure_pixels_loaded()
                    data_result = st.session_state["data_result"]
                    df_pixels = data_result["df_pixels"]
                    shapes = data_result["shapes"]

                    overlay_fig, _ = overlay_multi_channels(
                        df=df_pixels,
                        shapes=shapes,
                        image_id=config.image_id,
                        channels_cfg=[channel.to_dict() for channel in config.channels],
                        overlay_channels=config.overlay_channels or [channel.channel for channel in config.channels],
                        white_channel=config.white_channel,
                        white_weight=config.white_weight,
                        clip_hi=99.8,
                        pixel_size_um=config.pixel_size_um,
                        save_path=config.save_dir / "overlay.svg",
                    )
                    split_fig = plot_split_channels(
                        df=df_pixels,
                        shapes=shapes,
                        image_id=config.image_id,
                        channels_cfg=[channel.to_dict() for channel in config.channels],
                        pixel_size_um=config.pixel_size_um,
                        clip_hi=99.8,
                        save_path=config.save_dir / "split_channels.svg",
                    )
                    st.session_state["data_result"]["overlay_figure"] = overlay_fig
                    st.session_state["data_result"]["split_figure"] = split_fig
                st.success(f"Saved overlay.svg and split_channels.svg to {config.save_dir}")
            except Exception as exc:
                st.error(str(exc))

        data_result = st.session_state.get("data_result")
        if data_result and data_result.get("overlay_figure") is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Overlay")
                st.pyplot(data_result["overlay_figure"], clear_figure=False)
            with col2:
                st.markdown("#### Split channels")
                st.pyplot(data_result["split_figure"], clear_figure=False)


def render_nuclei_tab(tab):
    with tab:
        st.subheader("Nuclei segmentation")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is None:
            st.warning("Save the configuration first.")
            return

        channel_names = [channel.channel for channel in config.channels]
        default_nucleus = guess_nuclear_channel(channel_names) or (channel_names[0] if channel_names else None)
        if "nucleus_channel_ui" not in st.session_state and default_nucleus is not None:
            st.session_state["nucleus_channel_ui"] = default_nucleus

        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("NUCLEUS_CHANNEL", options=channel_names, key="nucleus_channel_ui")
            st.slider("MIN_DIAM_UM", min_value=2.0, max_value=20.0, value=6.0, step=0.5, key="min_diam_um_ui")
            st.slider("MAX_DIAM_UM", min_value=0.0, max_value=120.0, value=60.0, step=1.0, key="max_diam_um_ui")
            st.slider("TOPHAT_RADIUS_UM", min_value=0.0, max_value=8.0, value=2.0, step=0.5, key="tophat_radius_um_ui")
            st.slider("GAUSS_SIGMA_UM", min_value=0.0, max_value=3.0, value=0.5, step=0.1, key="gauss_sigma_um_ui")
        with col2:
            st.slider("LOCAL_WIN_UM", min_value=5.0, max_value=80.0, value=25.0, step=1.0, key="local_win_um_ui")
            st.slider("LOCAL_OFFSET", min_value=-0.20, max_value=0.20, value=-0.03, step=0.01, key="local_offset_ui")
            st.slider("H_MAXIMA_UM", min_value=0.05, max_value=5.0, value=0.25, step=0.05, key="h_maxima_um_ui")
            st.slider("SEED_MIN_DIST_UM", min_value=0.1, max_value=10.0, value=0.1, step=0.1, key="seed_min_dist_um_ui")
            st.slider("WATERSHED_COMPACTNESS", min_value=0.0, max_value=2.0, value=0.5, step=0.05, key="watershed_compactness_ui")
            st.slider("POST_RESPLIT_MULT", min_value=0.25, max_value=3.0, value=0.5, step=0.05, key="post_resplit_mult_ui")
        st.checkbox("Save outputs (unchecked = preview only)", value=True, key="save_nuclei_outputs_ui")

        if st.button("Run nuclei segmentation", type="primary", key="run_nuclei_btn"):
            try:
                with st.spinner("Running nuclei segmentation..."):
                    ensure_pixels_loaded()
                    data_result = st.session_state["data_result"]
                    params = NucleiParams(
                        nucleus_channel=st.session_state["nucleus_channel_ui"],
                        min_diam_um=float(st.session_state["min_diam_um_ui"]),
                        max_diam_um=float(st.session_state["max_diam_um_ui"]),
                        tophat_radius_um=float(st.session_state["tophat_radius_um_ui"]),
                        gauss_sigma_um=float(st.session_state["gauss_sigma_um_ui"]),
                        local_win_um=float(st.session_state["local_win_um_ui"]),
                        local_offset=float(st.session_state["local_offset_ui"]),
                        h_maxima_um=float(st.session_state["h_maxima_um_ui"]),
                        seed_min_dist_um=float(st.session_state["seed_min_dist_um_ui"]),
                        watershed_compactness=float(st.session_state["watershed_compactness_ui"]),
                        post_resplit_mult=float(st.session_state["post_resplit_mult_ui"]),
                    )
                    result = run_nuclei_segmentation(
                        df_pixels=data_result["df_pixels"],
                        shapes=data_result["shapes"],
                        image_id=config.image_id,
                        save_dir=config.save_dir,
                        pixel_size_um=config.pixel_size_um,
                        params=params,
                        save_outputs=bool(st.session_state["save_nuclei_outputs_ui"]),
                    )
                    st.session_state["nuclei_result"] = result
                    invalidate_after_nuclei_change()
                st.success(f"Nuclei segmentation finished. Outputs are in {config.save_dir}")
            except Exception as exc:
                st.error(str(exc))

        nuclei_result = st.session_state.get("nuclei_result")
        if nuclei_result is not None:
            st.metric("Segmented nuclei", int(nuclei_result["n_nuclei"]))
            st.pyplot(nuclei_result["figure"], clear_figure=False)
            st.markdown("#### nuclei_summary.csv preview")
            st.dataframe(nuclei_result["df_props"].head(20), use_container_width=True)


def render_celltypes_tab(tab):
    with tab:
        st.subheader("Cell-type definition")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is None:
            st.warning("Save the configuration first.")
            return

        channel_names = [channel.channel for channel in config.channels]
        marker_choices = marker_choices_for_ui(channel_names)
        token_map = token_mapping_for_ui(channel_names)

        ensure_celltype_items()

        st.write(f"Available markers: {', '.join(marker_choices)}")
        st.caption("Priority order matters: the first matching cell type wins; if none match, type #1 is used as fallback.")

        top_cols = st.columns([1, 1, 6])
        with top_cols[0]:
            if st.button("Add cell type", key="add_celltype_btn"):
                add_celltype_item()
                st.rerun()
        with top_cols[1]:
            if st.button("Save cell types", type="primary", key="save_celltypes_btn"):
                try:
                    cfg = save_celltype_cfg_from_widgets(channel_names)
                    st.success("Cell-type configuration saved.")
                    st.json(cfg)
                except Exception as exc:
                    st.error(str(exc))

        for idx, item in enumerate(st.session_state["celltype_items"]):
            uid = item["id"]
            default_name = item["default_name"]
            default_color = item["default_color"]

            title = st.session_state.get(f"ct_name_{uid}", default_name)
            with st.expander(f"{idx + 1}. {title}", expanded=True):
                header_cols = st.columns([2.2, 1.2, 1, 1, 1])
                with header_cols[0]:
                    st.text_input("Name", value=default_name, key=f"ct_name_{uid}")
                with header_cols[1]:
                    st.color_picker("Color", value=default_color, key=f"ct_color_{uid}")
                with header_cols[2]:
                    st.selectbox(
                        "Mode",
                        options=["simple", "expr"],
                        format_func=lambda v: "Simple logic" if v == "simple" else "Advanced expression",
                        key=f"ct_mode_{uid}",
                    )
                with header_cols[3]:
                    if st.button("↑", key=f"move_up_{uid}"):
                        move_celltype_item(uid, -1)
                        st.rerun()
                with header_cols[4]:
                    if st.button("↓", key=f"move_down_{uid}"):
                        move_celltype_item(uid, 1)
                        st.rerun()

                if st.button("Remove this type", key=f"remove_{uid}"):
                    remove_celltype_item(uid)
                    st.rerun()

                mode = st.session_state.get(f"ct_mode_{uid}", "simple")
                if mode == "simple":
                    st.multiselect(
                        "ALL positive markers (AND)",
                        options=marker_choices,
                        key=f"ct_all_pos_{uid}",
                    )
                    st.multiselect(
                        "ALL negative markers (AND)",
                        options=marker_choices,
                        key=f"ct_all_neg_{uid}",
                    )
                    st.number_input(
                        "Number of ANY-positive groups",
                        min_value=0,
                        max_value=8,
                        step=1,
                        key=f"ct_group_count_{uid}",
                    )
                    group_count = int(st.session_state.get(f"ct_group_count_{uid}", 0))
                    for group_idx in range(group_count):
                        st.multiselect(
                            f"Group {group_idx + 1}: at least one marker must be positive",
                            options=marker_choices,
                            key=f"ct_group_{uid}_{group_idx}",
                        )
                else:
                    token_help = "\n".join([f"- `{marker}` → `{token}`" for marker, token in token_map.items()])
                    st.markdown("Token mapping:\n" + token_help)
                    st.text_area(
                        "Expression (use AND/OR/NOT or & | ~ with parentheses)",
                        key=f"ct_expr_{uid}",
                        height=120,
                    )

        if st.session_state.get("celltype_cfg") is not None:
            st.markdown("#### Current saved CELLTYPE_CFG")
            st.json(st.session_state["celltype_cfg"])


def render_assignment_tab(tab):
    with tab:
        st.subheader("Assign cell types to nuclei and build masks")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is None:
            st.warning("Save the configuration first.")
            return
        if st.session_state.get("celltype_cfg") is None:
            st.warning("Save the cell-type configuration first.")
            return

        if st.button("Run cell-type assignment", type="primary", key="run_assignment_btn"):
            try:
                with st.spinner("Assigning marker positivity and cell types..."):
                    ensure_pixels_loaded()
                    labels = ensure_labels_available()
                    data_result = st.session_state["data_result"]
                    result = run_celltype_assignment(
                        folder=config.folder,
                        save_dir=config.save_dir,
                        pixel_size_um=config.pixel_size_um,
                        image_id=config.image_id,
                        channels_cfg=[channel.to_dict() for channel in config.channels],
                        celltype_cfg=st.session_state["celltype_cfg"],
                        labels=labels,
                        df_pixels=data_result["df_pixels"],
                        shapes=data_result["shapes"],
                        save_outputs=True,
                    )
                    st.session_state["assignment_result"] = result
                    invalidate_after_assignment_change()
                st.success(f"Cell-type assignment finished. Outputs are in {config.save_dir}")
            except Exception as exc:
                st.error(str(exc))

        assignment_result = st.session_state.get("assignment_result")
        if assignment_result is None:
            try:
                assignment_result = ensure_assignment_outputs_available()
            except Exception:
                assignment_result = None

        if assignment_result is not None:
            st.markdown("#### celltype_counts.csv")
            st.dataframe(assignment_result["counts"], use_container_width=True)
            if assignment_result.get("thresholds") is not None:
                with st.expander("Marker assignment thresholds"):
                    st.dataframe(assignment_result["thresholds"], use_container_width=True)
            if assignment_result.get("panel_figure") is not None:
                st.markdown("#### Cell-type panel")
                st.pyplot(assignment_result["panel_figure"], clear_figure=False)
            if assignment_result.get("split_figure") is not None:
                st.markdown("#### Split panels")
                st.pyplot(assignment_result["split_figure"], clear_figure=False)
            st.markdown("#### cells_summary.csv preview")
            st.dataframe(assignment_result["df_cells"].head(20), use_container_width=True)


def render_regions_tab(tab):
    with tab:
        st.subheader("Boundary / region segmentation")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is None:
            st.warning("Save the configuration first.")
            return
        if st.session_state.get("celltype_cfg") is None:
            st.warning("Save the cell-type configuration first.")
            return

        try:
            assignment_result = ensure_assignment_outputs_available()
        except Exception as exc:
            st.warning(str(exc))
            return

        celltype_names = [ct["name"] for ct in st.session_state["celltype_cfg"]]
        present_types = sorted(set(assignment_result["df_cells"]["celltype"].astype(str)))
        celltype_names = [name for name in celltype_names if name in present_types] or present_types

        st.multiselect(
            "Select one or more cell types to define region boundaries",
            options=celltype_names,
            default=celltype_names[:1],
            key="region_selected_types",
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.slider("Close (µm)", min_value=0.0, max_value=80.0, value=15.0, step=1.0, key="region_close_um")
        with col2:
            st.slider("Dilate (µm)", min_value=0.0, max_value=80.0, value=10.0, step=1.0, key="region_dilate_um")
        with col3:
            st.number_input("Min area (µm²)", min_value=0.0, value=20000.0, step=1000.0, key="region_min_area_um2")
        with col4:
            st.number_input("Min cells", min_value=1, value=5, step=1, key="region_min_cells")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.selectbox("Contour downsample", options=[1, 2, 4, 8], index=1, key="region_contour_ds")
        with col2:
            st.slider("Boundary line width", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="region_line_width")
        with col3:
            st.selectbox(
                "Boundary line style",
                options=["-", "--", "-.", ":"],
                format_func=lambda v: {"-": "Solid", "--": "Dashed", "-.": "Dash-dot", ":": "Dotted"}[v],
                key="region_line_style",
            )
        with col4:
            st.color_picker("Boundary color", value="#a1d99b", key="region_boundary_color")

        st.checkbox("Use each type's own color for the boundary", value=False, key="region_use_type_colors")

        if st.button("Run boundaries + counts", type="primary", key="run_region_btn"):
            try:
                selected_types = list(st.session_state.get("region_selected_types", []))
                if not selected_types:
                    raise RuntimeError("Please select at least one cell type.")
                params = RegionParams(
                    selected_types=selected_types,
                    close_um=float(st.session_state["region_close_um"]),
                    dilate_um=float(st.session_state["region_dilate_um"]),
                    min_area_um2=float(st.session_state["region_min_area_um2"]),
                    min_cells=int(st.session_state["region_min_cells"]),
                    contour_downsample=int(st.session_state["region_contour_ds"]),
                    line_width=float(st.session_state["region_line_width"]),
                    line_style=str(st.session_state["region_line_style"]),
                    boundary_color=str(st.session_state["region_boundary_color"]),
                    use_type_colors=bool(st.session_state["region_use_type_colors"]),
                )
                with st.spinner("Building region masks and counting cells..."):
                    result = run_region_boundary_analysis(
                        df_cells=assignment_result["df_cells"],
                        celltype_mask=assignment_result["celltype_mask"],
                        celltype_cfg=st.session_state["celltype_cfg"],
                        save_dir=config.save_dir,
                        pixel_size_um=config.pixel_size_um,
                        params=params,
                        save_outputs=True,
                    )
                    st.session_state["region_result"] = result
                st.success(f"Region analysis finished. Outputs are in {config.save_dir}")
            except Exception as exc:
                st.error(str(exc))

        region_result = st.session_state.get("region_result")
        if region_result is not None:
            st.pyplot(region_result["figure"], clear_figure=False)
            st.markdown("#### celltype_counts_by_region preview")
            st.dataframe(region_result["counts_by_region"], use_container_width=True)


def render_distance_tab(tab):
    with tab:
        st.subheader("Distance analyses")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is None:
            st.warning("Save the configuration first.")
            return
        if st.session_state.get("celltype_cfg") is None:
            st.warning("Save the cell-type configuration first.")
            return

        try:
            assignment_result = ensure_assignment_outputs_available()
        except Exception as exc:
            st.warning(str(exc))
            return

        df_cells = assignment_result["df_cells"]
        celltype_mask = assignment_result["celltype_mask"]

        present_types = sorted(set(df_cells["celltype"].astype(str)))
        nn_tab, boundary_tab = st.tabs(["Nearest-neighbor distances", "Cell-to-boundary distances"])

        with nn_tab:
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Target cell type", options=present_types, key="nn_target_type")
            with col2:
                st.multiselect(
                    "Query cell types",
                    options=present_types,
                    default=present_types[:1],
                    key="nn_query_types",
                )
            if st.button("Compute nearest-neighbor distances", type="primary", key="run_nn_btn"):
                try:
                    queries = list(st.session_state.get("nn_query_types", []))
                    if not queries:
                        raise RuntimeError("Select at least one query cell type.")
                    with st.spinner("Computing nearest-neighbor distances..."):
                        result = run_nearest_neighbor_analysis(
                            df_cells=df_cells,
                            celltype_cfg=st.session_state["celltype_cfg"],
                            save_dir=config.save_dir,
                            pixel_size_um=config.pixel_size_um,
                            target_type=st.session_state["nn_target_type"],
                            query_types=queries,
                            save_outputs=True,
                        )
                        st.session_state["nn_result"] = result
                    st.success("Nearest-neighbor distance analysis finished.")
                except Exception as exc:
                    st.error(str(exc))

            nn_result = st.session_state.get("nn_result")
            if nn_result is not None:
                st.pyplot(nn_result["figure"], clear_figure=False)
                st.markdown("#### Distances preview")
                st.dataframe(nn_result["distances"].head(20), use_container_width=True)
                if not nn_result["ttests"].empty:
                    st.markdown("#### Paired t-tests")
                    st.dataframe(nn_result["ttests"], use_container_width=True)

        with boundary_tab:
            boundary_candidates = discover_boundary_masks(
                save_dir=config.save_dir,
                celltype_cfg=st.session_state["celltype_cfg"],
                df_cells=df_cells,
            )
            if not boundary_candidates:
                st.info("No boundary region masks were found yet. Run the region analysis step first.")
            else:
                boundary_labels = [f"{name} — {path.name}" for name, path in boundary_candidates]
                label_to_path = {label: path for label, (_, path) in zip(boundary_labels, boundary_candidates)}
                label_to_name = {label: name for label, (name, _) in zip(boundary_labels, boundary_candidates)}

                col1, col2, col3 = st.columns([2, 2, 1.5])
                with col1:
                    st.selectbox("Boundary mask", options=boundary_labels, key="boundary_mask_label")
                with col2:
                    st.multiselect(
                        "Query cell types",
                        options=present_types,
                        default=present_types[:1],
                        key="boundary_query_types",
                    )
                with col3:
                    st.selectbox(
                        "Filter",
                        options=["all", "inside", "outside"],
                        format_func=lambda v: {
                            "all": "All cells",
                            "inside": "Only cells inside region",
                            "outside": "Only cells outside region",
                        }[v],
                        key="boundary_region_filter",
                    )

                if st.button("Compute boundary distances", type="primary", key="run_boundary_dist_btn"):
                    try:
                        queries = list(st.session_state.get("boundary_query_types", []))
                        if not queries:
                            raise RuntimeError("Select at least one query cell type.")
                        selected_label = st.session_state["boundary_mask_label"]
                        boundary_path = label_to_path[selected_label]
                        boundary_name = label_to_name[selected_label]
                        with st.spinner("Computing distances to boundary..."):
                            result = run_boundary_distance_analysis(
                                df_cells=df_cells,
                                celltype_cfg=st.session_state["celltype_cfg"],
                                celltype_mask=celltype_mask,
                                save_dir=config.save_dir,
                                pixel_size_um=config.pixel_size_um,
                                boundary_mask_path=boundary_path,
                                boundary_name=boundary_name,
                                query_types=queries,
                                region_filter=st.session_state["boundary_region_filter"],
                                save_outputs=True,
                            )
                            st.session_state["boundary_result"] = result
                        st.success("Boundary distance analysis finished.")
                    except Exception as exc:
                        st.error(str(exc))

                boundary_result = st.session_state.get("boundary_result")
                if boundary_result is not None:
                    st.pyplot(boundary_result["figure"], clear_figure=False)
                    st.markdown("#### Distances preview")
                    st.dataframe(boundary_result["distances"].head(20), use_container_width=True)


def render_outputs_tab(tab):
    with tab:
        st.subheader("Outputs")
        config: PipelineConfig | None = st.session_state.get("config")
        if config is None:
            st.warning("Save the configuration first.")
            return

        st.write(f"Output directory: `{config.save_dir}`")
        output_rows = list_output_files(config.save_dir)
        if not output_rows:
            st.info("No outputs have been generated yet.")
            return

        df_files = pd.DataFrame(output_rows)
        st.dataframe(df_files, use_container_width=True)

        zip_bytes = zip_directory_bytes(config.save_dir)
        if zip_bytes:
            st.download_button(
                "Download current outputs as ZIP",
                data=zip_bytes,
                file_name=f"TME_spatial_outputs_{st.session_state['session_id']}.zip",
                mime="application/zip",
            )

        if config.input_mode == "local":
            st.caption("Because you are in local-folder mode, files are also already saved directly to the selected folder/outs.")
        else:
            st.caption("Because you are in upload mode, the files are stored in a temporary session folder and can be downloaded as a ZIP.")


def main():
    init_state()
    render_sidebar()

    st.title("TME Spatial — Streamlit app")
    st.write(
        "This app packages your notebook pipeline into a Streamlit interface while keeping the original "
        "analysis logic as close as possible. Input files are the ImageJ-exported CSV/TXT text images "
        "containing coordinates/intensity grids."
    )

    tabs = st.tabs(
        [
            "1. Inputs & config",
            "2. Overlay preview",
            "3. Nuclei segmentation",
            "4. Cell types",
            "5. Cell-type assignment",
            "6. Region analysis",
            "7. Distance analysis",
            "8. Outputs",
        ]
    )
    render_config_tab(tabs[0])
    render_overlay_tab(tabs[1])
    render_nuclei_tab(tabs[2])
    render_celltypes_tab(tabs[3])
    render_assignment_tab(tabs[4])
    render_regions_tab(tabs[5])
    render_distance_tab(tabs[6])
    render_outputs_tab(tabs[7])


if __name__ == "__main__":
    main()
