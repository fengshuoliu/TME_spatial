# TME Spatial Streamlit App

This repo packages the uploaded notebook pipeline into a Streamlit application while keeping the original analysis logic as close as possible.

The expected inputs are the ImageJ-exported text images (`.csv` or `.txt`) that store per-pixel intensity grids. The app supports:

- multi-channel input configuration
- overlay and split-channel figure generation
- nuclei segmentation
- interactive cell-type definition
- cell-type assignment
- region / boundary analysis
- nearest-neighbor distance analysis
- cell-to-boundary distance analysis

## Important deployment note

   The app only accept upload mode, but all the analysis is computated locally. This keeps your original folder-based workflow for local use, while still making the app deployable for other users.

   **Upload CSV/TXT files**  
   Use this when the app is publicly hosted. Uploaded files are stored in a temporary session workspace, and outputs can be downloaded as a ZIP from the app.


## Installation

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```


## Run locally

From the repo root:

```bash
streamlit run app.py
```

## App workflow

### 1. Upload CSV/TXT files

Then:

- set the number of channels
- choose each file
- set each marker name
- set each color
- enter pixel size:
  - `x (µm)` and `x (px)`
  - `y (µm)` and `y (px)`
- choose overlay channels and optional white overlay
- click **Save configuration**

The app writes `config.json` into the output directory.

### 2. Overlay preview

Click **Load inputs and generate overlay**.

Saved outputs:

- `overlay.svg`
- `split_channels.svg`

### 3. Nuclei segmentation

Choose the nuclear channel and adjust the tuning sliders if needed.

Click **Run nuclei segmentation**.

Saved outputs:

- `nuclei_labels_uint16.tiff`
- `nuclei_summary.csv`
- `nuclei_boundaries.json`
- `nuclei_segmentation_panel.svg`
- `nuclei_segmentation_panel.tiff`
- `nuclei_params.json`

### 4. Cell types

Define cell types with either:

- **Simple logic**
  - ALL positive markers
  - ALL negative markers
  - any-positive groups
- **Advanced expression**
  - boolean expressions using marker tokens

Priority order is **top to bottom**.  
The first matching type wins.  
If nothing matches, the app uses the first cell type as the fallback, matching the notebook behavior.

Click **Save cell types**.

Saved output:

- `celltype_config.json`

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

### 6. Region analysis

Select one or more cell types to define regions and set:

- close radius
- dilate radius
- minimum region area
- minimum cells
- contour downsample
- boundary style

Click **Run boundaries + counts**.

Saved outputs:

- `<type>_region_mask_uint8.tiff`
- `celltypes_with_boundaries__<types>.svg`
- `celltypes_with_boundaries__<types>.tiff`
- `celltype_counts_by_region__<types>.csv`
- `cell_region_assignments__<types>.csv`
- `region_params__<types>.json`

### 7. Distance analysis

Two analyses are available.

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

### 8. Outputs

The app lists all generated files and lets you download them as a ZIP.

## Expected input format

Each input file should be a 2D intensity grid exported from ImageJ as text, such as:

- comma-separated (`.csv`)
- tab-separated (`.txt`)
- semicolon-separated
- whitespace-separated

The loader auto-detects the delimiter.

## Testing locally before GitHub upload

A safe local test sequence is:

1. Run the app locally with `streamlit run app.py`
2. Use **Local folder path** mode
3. Point the app at a folder containing your exported channel files
4. Click through the tabs in order
5. Confirm that outputs appear in `<folder>/outs`

## Uploading into your GitHub repo

You said you want to upload this into:

```text
https://github.com/fengshuoliu/TME_spatial/tree/main
```

Use the repo root (not the `/tree/main` page) for your Git operations. A simple workflow is:

```bash
git clone https://github.com/fengshuoliu/TME_spatial.git
cd TME_spatial
```

Then copy the generated files from this bundle into that repo, for example:

```bash
cp -R /path/to/TME_spatial_streamlit_app/* .
cp -R /path/to/TME_spatial_streamlit_app/.streamlit ./
```

Then commit and push:

```bash
git add .
git commit -m "Add Streamlit app packaging for TME spatial pipeline"
git push
```

## Streamlit deployment

After the files are in GitHub, point Streamlit at:

- **main file**: `app.py`

For public deployment, tell users to use **Upload CSV/TXT files** mode.

For local/lab-internal use on the same machine as the data, use **Local folder path** mode.

## Notebook references

The original uploaded notebook is included here:

- `notebooks/original_pipeline_notebook.ipynb`

A small example notebook showing how to call the refactored modules is also included:

- `notebooks/pipeline_demo.ipynb`

## Notes

- The app preserves the notebook's default behavior where possible.
- Large images can take time and memory, especially during cell-type assignment and region analysis.
- If performance is slow, use fewer channels, smaller test data, or install `numba`.
- If you retune nuclei segmentation, rerun the downstream steps so all outputs stay consistent.
