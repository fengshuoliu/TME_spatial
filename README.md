# TME Spatial Streamlit App

## Project website

[https://fengshuoliu.github.io/TME_spatial/](https://fengshuoliu.github.io/TME_spatial/)

A Streamlit interface for the TME spatial image-analysis pipeline, refactored from the original notebook while keeping the analysis logic as close as possible to the notebook workflow.

The app is designed for **ImageJ-exported text images** (`.csv` or `.txt`) that store **2D per-pixel intensity grids**. It supports:

- multi-channel input configuration
- overlay and split-channel figure generation
- nuclei segmentation
- interactive cell-type definition
- cell-type assignment
- region / boundary analysis
- nearest-neighbor distance analysis
- cell-to-boundary distance analysis
- downloadable session outputs as a ZIP archive

---

## Table of contents

- [Overview](#overview)
- [Current input model](#current-input-model)
- [Repository structure](#repository-structure)
- [Installation](#installation)
  - [macOS / Linux](#macos--linux)
  - [Windows PowerShell quick start](#windows-powershell-quick-start)
- [Run the app](#run-the-app)
- [App workflow](#app-workflow)
- [Parameter guide](#parameter-guide)
- [Expected input format](#expected-input-format)
- [Generated outputs](#generated-outputs)
- [Troubleshooting](#troubleshooting)
- [Notebook references](#notebook-references)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository packages the original notebook pipeline into a Streamlit application so that users can run the workflow through a web interface instead of manually stepping through notebook cells.

The implementation is intended to preserve the original notebook behavior wherever practical, including:

- channel-based image loading from text-image grids
- notebook-style overlay generation
- nuclei segmentation followed by per-cell marker assignment
- top-to-bottom cell-type rule priority
- region mask generation from one or more selected cell types
- downstream nearest-neighbor and boundary-distance analyses

---

## Current input model

The current app is **upload-only**.

Users upload ImageJ-exported `.csv` / `.txt` intensity-grid files directly into the app. The app stores those uploaded files in a **temporary per-session workspace**, performs the analysis there, and allows the generated outputs to be downloaded as a ZIP file from the **Outputs** tab.

This design makes the app easier to run across different machines and easier to deploy through Streamlit.

---

## Repository structure

```text
TME_spatial_streamlit_app/
├── app.py
├── src/
│   └── tme_spatial/
│       ├── __init__.py
│       ├── io.py
│       ├── models.py
│       ├── visualization.py
│       ├── nuclei_segmentation.py
│       ├── celltype_assignment.py
│       ├── region_analysis.py
│       └── distance_analysis.py
├── notebooks/
│   ├── original_pipeline_notebook.ipynb
│   └── pipeline_demo.ipynb
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
└── .streamlit/
    └── config.toml
```

---

## Installation

### Recommended Python version

Use **Python 3.11** when possible.

---

### macOS / Linux

From the repository root:

#### 1. Create an environment through conda, and activate it

```bash
conda create -n tme_spatial python=3.11 -y
conda activate tme_spatial
```

#### 2. Move to the repository folder

```bash
cd ~/TME_spatial
```
Replace `~/TME_spatial` with the actual path to your repository.

#### 3. Install dependencies

```bash
pip install -r requirement.txt
```

#### 4. Launch the app

```bash
streamlit run app.py
```

---

### Windows PowerShell quick start

If you are using Windows without Linux / WSL, you can install and run the app entirely through **PowerShell**.

#### 1. Install Python

A simple Windows-native route is:

- install Python through **Python Install Manager** from the Microsoft Store
- open **PowerShell**
- install Python 3.11

```powershell
py install 3.11
py -3.11 --version
```

#### 2. Create a virtual environment

```powershell
py -3.11 -m venv tme_spatial
```

#### 3. Move to the repository folder

```powershell
cd D:\TME_spatial
```

Replace `D:\TME_spatial` with the actual path to your repository.

#### 4. Allow script execution for the current PowerShell session

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### 5. Activate the environment

```powershell
tme_spatial\Scripts\Activate.ps1
```

#### 6. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements.txt
```

#### 7. Launch the app

```powershell
python -m streamlit run app.py
```

If the `py` launcher is unavailable on your machine, you may need to call Python using its full path.

Example:

```text
C:\Users\xuzha\AppData\Local\Python\bin\python.exe
```

---

## Run the app

From the repository root:

```bash
python -m streamlit run app.py
```

After launch, Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

---

## App workflow

### 1. Inputs & config

Upload the ImageJ-exported `.csv` / `.txt` files.

Then configure:

- number of channels
- channel-to-file assignments
- marker names
- display colors
- pixel size calibration:
  - `x (µm)` and `x (px)`
  - `y (µm)` and `y (px)`
- overlay channels
- optional white overlay channel and weight

Click **Save configuration**.

Saved output:

- `config.json`

---

### 2. Overlay preview

Click **Load inputs and generate overlay**.

Saved outputs:

- `overlay.svg`
- `split_channels.svg`

---

### 3. Nuclei segmentation

Choose the nuclear channel and adjust segmentation parameters as needed.

Click **Run nuclei segmentation**.

Saved outputs:

- `nuclei_labels_uint16.tiff`
- `nuclei_summary.csv`
- `nuclei_boundaries.json`
- `nuclei_segmentation_panel.svg`
- `nuclei_segmentation_panel.tiff`
- `nuclei_params.json`

---

### 4. Cell types

Define cell types using either:

- **Simple logic**
  - all-positive markers
  - all-negative markers
  - any-positive groups
- **Advanced expression**
  - boolean logic using marker tokens

Priority order is **top to bottom**. The first matching rule wins.

If a cell does not match any rule, the app uses the **first cell type** as fallback, matching the notebook behavior.

Click **Save cell types**.

Saved output:

- `celltype_config.json`

---

### 5. Cell-type assignment

Click **Run cell-type assignment**.

Saved outputs:

- `marker_assign_<MARKER>_uint16.tiff`
- `marker_assignment_thresholds.csv`
- `celltypes_mask_uint16.tiff`
- `cells_summary.csv`
- `celltype_counts.csv`
- `celltypes_panel.svg`
- `celltypes_panel.tiff`
- `celltypes_split_panels.svg`
- `celltypes_split_panels.tiff`

---

### 6. Region analysis

Choose one or more cell types to define the region boundary and set the region parameters.

Click **Run boundaries + counts**.

Saved outputs:

- `<type>_region_mask_uint8.tiff`
- `celltypes_with_boundaries__<types>.svg`
- `celltypes_with_boundaries__<types>.tiff`
- `celltype_counts_by_region__<types>.csv`
- `cell_region_assignments__<types>.csv`
- `region_params__<types>.json`

---

### 7. Distance analysis

Two downstream analyses are available.

#### Nearest-neighbor distances

Saved outputs:

- `nearest_neighbor_distances__<...>.csv`
- `nearest_neighbor_distances__<...>.svg`
- `nearest_neighbor_distances__<...>.png`

#### Cell-to-boundary distances

Saved outputs:

- `dist_to_boundary__<...>.csv`
- `dist_to_boundary__<...>.svg`
- `dist_to_boundary__<...>.png`

---

### 8. Outputs

The app lists all generated outputs from the current session and allows them to be downloaded as a ZIP archive.

---

## Parameter guide

The tables below summarize the main user-adjustable parameters in each major step, what each parameter controls, and what typically happens when the value is increased or decreased.

### 1. Inputs & configuration

| Parameter | What it controls | Larger / more | Smaller / less | Practical note |
|---|---|---:|---:|---|
| `Number of channels` | How many uploaded files are configured as channels | More channels to display and analyze; richer overlays; more memory/runtime | Simpler setup and faster testing | Set this to the true number of marker grids you want to include |
| `Channel file` | Which uploaded file is assigned to a channel slot | — | — | Must match the correct marker image |
| `Marker` | Marker name used in plots and cell-type logic | — | — | Use stable names because later rules depend on them |
| `Color` | Visualization color for the channel | — | — | Affects figures only |
| `x (µm)` | Physical width corresponding to `x (px)` | Increases µm-per-pixel in **x** | Decreases µm-per-pixel in **x** | Affects scale-dependent outputs |
| `x (px)` | Pixel width corresponding to `x (µm)` | Decreases µm-per-pixel in **x** | Increases µm-per-pixel in **x** | Incorrect values will distort all scale-based calculations |
| `y (µm)` | Physical height corresponding to `y (px)` | Increases µm-per-pixel in **y** | Decreases µm-per-pixel in **y** | Same role as `x (µm)` for y |
| `y (px)` | Pixel height corresponding to `y (µm)` | Decreases µm-per-pixel in **y** | Increases µm-per-pixel in **y** | Same role as `x (px)` for y |
| `Overlay channels` | Channels combined into the composite overlay | Richer composite, but can become visually crowded | Cleaner overlay | Visualization only |
| `White overlay channel` | Optional channel displayed as white | — | — | Useful for structural context |
| `White overlay weight` | Strength of the white overlay contribution | Stronger white influence, possibly washing out colors | Weaker white contribution | Visualization only |

### 2. Overlay preview

This step has no additional analysis-tuning parameters in the current app.

### 3. Nuclei segmentation

| Parameter | What it controls | Larger / more | Smaller / less | Practical note |
|---|---|---:|---:|---|
| `NUCLEUS_CHANNEL` | Channel used for nuclear segmentation | — | — | Choose the true nuclear stain, such as `DAPI` |
| `MIN_DIAM_UM` | Minimum expected nucleus diameter | Removes more small debris/noise; may miss tiny nuclei | Keeps smaller nuclei; may admit noise | Increase if you see many tiny false positives |
| `MAX_DIAM_UM` | Maximum expected nucleus diameter | Allows larger nuclei/merged blobs | Removes more large blobs; may exclude real large nuclei | Decrease if large merged objects remain |
| `TOPHAT_RADIUS_UM` | Background-correction radius | Stronger removal of broad background | Milder background correction | Useful for uneven nuclear background |
| `GAUSS_SIGMA_UM` | Gaussian smoothing before thresholding | More smoothing; less noise; more risk of merging neighbors | Sharper image; more noise sensitivity | Increase if segmentation is noisy |
| `LOCAL_WIN_UM` | Window size for local thresholding | More global threshold behavior | More local sensitivity | Larger values help with smooth illumination gradients |
| `LOCAL_OFFSET` | Offset used in local thresholding | Lower threshold, larger masks, more dim nuclei detected, but more background | Higher threshold, tighter masks, cleaner segmentation, but more missed dim nuclei | One of the most sensitive parameters |
| `H_MAXIMA_UM` | Peak-strength requirement for seed detection | Fewer seeds; less splitting | More seeds; more aggressive splitting | Decrease if touching nuclei remain merged |
| `SEED_MIN_DIST_UM` | Minimum spacing between seeds | Fewer, farther-apart seeds | More, closer seeds | Increase if nuclei are fragmented too much |
| `WATERSHED_COMPACTNESS` | Shape regularization during watershed | More compact regions | Regions follow local gradients more closely | Affects boundary geometry |
| `POST_RESPLIT_MULT` | Threshold for second-pass splitting of large objects | Fewer objects resplit | More objects resplit | Decrease if merged nuclei persist |

### 4. Cell types

| Parameter | What it controls | Larger / more | Smaller / less | Practical note |
|---|---|---:|---:|---|
| `Name` | Cell-type label used in outputs | — | — | Use biologically interpretable names |
| `Color` | Visualization color for the cell type | — | — | Affects plots only |
| `Mode` | Logic style: `simple` or `expr` | — | — | Use `simple` for most panels and `expr` for complex logic |
| `ALL positive markers (AND)` | Markers that must all be positive | Stricter definition; fewer cells match | Broader definition; more cells match | Good for high-confidence phenotypes |
| `ALL negative markers (AND)` | Markers that must all be negative | Stricter exclusion; fewer cells match | Broader exclusion | Useful for removing confounding lineages |
| `Number of ANY-positive groups` | Number of OR-groups that must each be satisfied | More groups = stricter overall rule | Fewer groups = broader rule | Groups are combined by AND |
| `Group i markers` | Marker list inside one OR-group | More markers in a group = easier for that group to pass | Fewer markers in a group = stricter group | Within a group, any one positive marker is sufficient |
| `Expression` | Advanced boolean rule using marker tokens | More clauses usually make the rule stricter | Fewer clauses usually make it broader | Use parentheses to make precedence explicit |
| `Priority order` | Match order from top to bottom | Moving a type higher makes it capture ambiguous cells earlier | Moving it lower makes it lose ties to higher-priority types | The first matching type wins |

### 5. Cell-type assignment

This step has no additional user-tuned analysis parameters in the current app. It applies the saved cell-type logic to the segmented nuclei.

### 6. Region analysis

| Parameter | What it controls | Larger / more | Smaller / less | Practical note |
|---|---|---:|---:|---|
| `Selected cell type(s)` | Which cell types define the region | Broader/composite region definition | More specific region definition | Choose the niche-defining populations |
| `Close (µm)` | Morphological closing radius | Bridges wider gaps; merges nearby patches | Preserves more gaps | Increase if the region looks fragmented |
| `Dilate (µm)` | Region expansion radius | Larger region footprint | Tighter region footprint | Increase for a broader niche definition |
| `Min area (µm²)` | Minimum region area to keep | Removes more small regions/noise | Keeps smaller regions | Increase if many tiny patches survive |
| `Min cells` | Minimum cells required in a region | Keeps denser, more robust regions | Keeps sparse regions | Increase if biologically meaningful regions should be cell-dense |
| `Contour downsample` | Downsampling for plotted contours | Coarser / faster contours | Finer / more detailed contours | Mainly affects display |
| `Boundary line width` | Thickness of plotted boundary | Thicker boundary line | Thinner boundary line | Display only |
| `Boundary line style` | Style of plotted boundary | — | — | Display only |
| `Boundary color` | Color of plotted boundary | — | — | Display only |
| `Use each type's own color` | Whether boundary color follows cell type | More visually linked to selected cell types | Uses one shared boundary color | Display only |

### 7. Distance analysis

#### Nearest-neighbor distances

| Parameter | What it controls | Larger / more | Smaller / less | Practical note |
|---|---|---:|---:|---|
| `Target cell type` | Reference population for nearest-neighbor calculation | — | — | Distances are measured from query cells to their nearest target cell |
| `Query cell types` | Cell types being compared to the target | More groups and comparisons | Simpler comparison | Use a focused set for easier interpretation |

#### Cell-to-boundary distances

| Parameter | What it controls | Larger / more | Smaller / less | Practical note |
|---|---|---:|---:|---|
| `Boundary mask` | Region boundary used as the distance reference | — | — | Usually generated in Step 6 |
| `Query cell types` | Cell types measured relative to the boundary | More groups and comparisons | Simpler output | Good for comparing immune/stromal/tumor localization |
| `Filter` | Whether to use all cells, only inside-region cells, or only outside-region cells | `all` gives the broadest summary | `inside` / `outside` gives more specific subsets | Use for intraregional vs extraregional comparisons |

### 8. Outputs

This step has no tuning parameters. It lists all generated files and provides ZIP download of the current session outputs.

---

## Expected input format

Each input file should be a **2D numeric intensity grid** exported from ImageJ as text.

Supported text-grid formats include:

- comma-separated (`.csv`)
- tab-separated (`.txt`)
- semicolon-separated
- whitespace-separated

The loader attempts to auto-detect the delimiter.

### Important input notes

- Each uploaded file should represent **one 2D image channel**.
- Do **not** upload summary tables, spreadsheets, or downstream result CSV files as channel inputs.
- Avoid opening and re-saving channel files in Excel or spreadsheet software unless you are certain the grid format is preserved.

---

## Generated outputs

Depending on which steps you run, the app can generate:

- configuration files (`.json`)
- summary tables (`.csv`)
- segmentation masks (`.tiff`)
- overlay and segmentation figures (`.svg`, `.tiff`, `.png`)
- nearest-neighbor and boundary-distance result tables and figures

All outputs for the current session can be downloaded from the **Outputs** tab as a ZIP archive.

---

## Troubleshooting

### 1. "The scientific Python stack for this app did not import correctly"

This usually means the Python environment is missing one or more required packages, or the wrong environment is active.

Try:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Then test imports directly:

```bash
python -c "import numpy, pandas, matplotlib, scipy, skimage, streamlit; print('Python stack OK')"
```

---

### 2. `streamlit` is not recognized

Use:

```bash
python -m streamlit run app.py
```

instead of:

```bash
streamlit run app.py
```

---

### 3. CSV / TXT parsing errors during overlay loading

If the app fails while loading an uploaded channel file:

- confirm the file is a true ImageJ text-image export
- confirm it contains a rectangular numeric grid
- confirm it was not re-saved in a spreadsheet tool with altered delimiters
- make sure you uploaded channel-image files, not summary result tables

---

### 4. App updated, but browser still shows old behavior

If you changed `app.py` or `src/tme_spatial/io.py` and the browser still shows the old version:

1. stop Streamlit fully
2. restart it from the repository root
3. refresh the browser

---

### 5. Performance is slow or memory usage is high

Large images and multi-step analyses can be memory-intensive.

To reduce load:

- test on fewer channels first
- use smaller pilot datasets when tuning parameters
- avoid running unnecessary downstream steps during debugging

---

## Notebook references

The repository includes:

- `notebooks/original_pipeline_notebook.ipynb` — the original notebook used as the source workflow
- `notebooks/pipeline_demo.ipynb` — a small example notebook showing how to call the refactored modules

---

## Citation

If you use this code in a publication or presentation, please cite the repository and include the project metadata from `CITATION.cff`.

---

## License

See `LICENSE` for reuse terms.
