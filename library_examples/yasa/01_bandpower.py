import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import mne
    import yasa

    from mne.channels import make_standard_montage
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    return eegbci, make_standard_montage, mne, mo, plt, read_raw_edf, yasa


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # YASA: Band Power

    This example demonstrates EEG band-power extraction using
    **YASA (`bandpower`)** with an MNE `Raw` object.

    The workflow follows the EEG Feature Atlas structure:

    **EEG → Preparation → Method → Feature → Numerical Output
    → Visualization → Interpretation**

    We will:

    1. download a public EEG recording;
    2. load the EDF file with MNE;
    3. standardize EEG channel names;
    4. attach standard electrode positions;
    5. calculate relative and absolute band power with YASA;
    6. inspect the numerical output;
    7. visualize average band power;
    8. visualize spatial band-power patterns;
    9. interpret the feature carefully;
    10. summarize the feature for the Atlas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Library / Method / Feature

    | Field | Value |
    |---|---|
    | Library | YASA |
    | Function | `yasa.bandpower()` |
    | Method | Welch power spectral density |
    | Feature | EEG band power |
    | Domain | Spectral + Spatial |
    | Input | MNE `Raw` |
    | Output | Channel × Band DataFrame |
    | Visualization | Bar chart + scalp topomaps |

    **Important distinction:**

    - Welch = spectral estimation method
    - PSD = spectral representation
    - Band power = feature extracted from the PSD
    - Bar chart / Topomap = visualization

    The domain is listed as **Spectral + Spatial** rather than
    Spectral alone: the feature itself (band power) is a spectral
    quantity, but this example also represents it spatially, one
    scalp topomap per band, the same reasoning used for the
    MNE PSD example earlier in the Atlas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset

    We use the public **EEG Motor Movement/Imagery Dataset
    (EEGBCI)** distributed through PhysioNet.

    The dataset contains 64-channel EEG recordings from multiple
    subjects and experimental runs.

    **Example recording:**

    - Subject: 1
    - Run: 2
    - Condition: baseline, eyes closed
    - Recording type: continuous EEG

    Run 2 corresponds to the eyes-closed baseline according to
    the MNE EEGBCI dataset documentation.

    The data are downloaded automatically by MNE if they are not
    already available locally.
    """)
    return


@app.cell
def _(eegbci):
    subject = 1
    run = 2

    eeg_files = eegbci.load_data(
        subjects=subject,
        runs=[run],
    )

    eeg_files
    return eeg_files, run, subject


@app.cell
def _(eeg_files, eegbci, make_standard_montage, read_raw_edf):
    raw = read_raw_edf(
        eeg_files[0],
        preload=True,
        verbose=False,
    )

    # Preprocessing step 1:
    # Standardize EEGBCI channel names to MNE-compatible names.
    eegbci.standardize(raw)

    # Preprocessing step 2:
    # Attach standard 10-05 electrode positions.
    _montage = make_standard_montage("standard_1005")
    raw.set_montage(_montage)

    # No filtering, resampling, epoching, ICA, or artifact removal
    # is applied in this minimal example.
    raw
    return (raw,)


@app.cell(hide_code=True)
def _(mo, raw, run, subject):
    mo.md(f"""
    ## Recording information

    | Field | Value |
    |---|---|
    | Subject | {subject} |
    | Run | {run} |
    | Channels | {len(raw.ch_names)} |
    | Sampling frequency | {raw.info["sfreq"]:.1f} Hz |
    | Duration | {raw.duration:.1f} seconds |

    ### Preparation

    The following preparation steps are explicit:

    - EEGBCI channel-name standardization
    - standard 10-05 montage assignment

    No filtering, resampling, epoching, ICA, or artifact removal
    is performed.

    This is intentional: the example focuses on the YASA feature
    rather than on a complete EEG preprocessing pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Band Power

    Band power measures how much spectral power is contained
    within a predefined frequency range.

    We use four commonly used EEG frequency bands:

    - **Theta:** 4–8 Hz
    - **Alpha:** 8–12 Hz
    - **Beta:** 12–30 Hz
    - **Gamma:** 30–45 Hz

    YASA estimates the power spectrum using Welch's method and
    integrates power within the requested frequency bands.

    With `relative=True`, YASA normalizes each band's power by the
    total power between the minimum and maximum frequencies
    specified by the requested bands.

    For these bands, the normalization range is therefore:

    **4–45 Hz**

    This means relative power depends on the selected band
    definitions. Relative-power values should therefore not be
    compared directly with results obtained using a different
    normalization range.

    YASA also provides total absolute power in the
    `TotalAbsPow` column. When `relative=False` is used, the
    returned band values represent non-normalized absolute power.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature parameters

    The important parameters are specified explicitly rather than
    relying on hidden defaults:

    - `relative=True` – return relative band power
    - `win_sec=4` – 4-second Welch windows
    - `bandpass=False` – do not apply an additional bandpass filter
    - `average="median"` – Welch averaging method
    - `window="hamming"` – Welch window function

    This makes the feature extraction step easier to reproduce.
    """)
    return


@app.cell
def _(raw, yasa):
    bands = [
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (12, 30, "Beta"),
        (30, 45, "Gamma"),
    ]

    bandpower_df = yasa.bandpower(
        raw,
        bands=bands,
        relative=True,
        win_sec=4,
        bandpass=False,
        kwargs_welch={
            "average": "median",
            "window": "hamming",
        },
    )

    bandpower_df
    return bandpower_df, bands


@app.cell
def _(bandpower_df, bands):
    band_names = [band[2] for band in bands]

    bandpower_shape = bandpower_df[band_names].shape
    bandpower_columns = list(bandpower_df.columns)

    (
        bandpower_shape,
        bandpower_columns,
    )
    return band_names, bandpower_columns, bandpower_shape


@app.cell(hide_code=True)
def _(band_names, bandpower_columns, bandpower_shape, mo):
    mo.md(f"""
    ## Numerical Output

    The YASA result is a `pandas.DataFrame`.

    - **Rows:** {bandpower_shape[0]} EEG channels
    - **Band-feature columns:** {", ".join(band_names)}
    - **Additional output:** `TotalAbsPow`

    **Output structure:**

    `Channel × Band`

    Each row represents one EEG channel.

    Each band column contains the relative band-power value
    calculated for that channel.

    Available columns:

    `{", ".join(bandpower_columns)}`
    """)
    return


@app.cell
def _(band_names, bandpower_df, plt):
    _band_means = bandpower_df[band_names].mean()

    _bar_figure, _bar_ax = plt.subplots(figsize=(7, 4))

    _bar_ax.bar(
        band_names,
        _band_means.values,
    )

    _bar_ax.set_xlabel("Frequency band")
    _bar_ax.set_ylabel("Mean relative power")
    _bar_ax.set_title("Average Relative Band Power Across EEG Channels")

    _bar_ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.close(_bar_figure)

    _bar_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatial Visualization

    Because the EEG channels have standard electrode positions,
    the channel-level band-power values can be visualized as
    scalp topomaps.

    Each map shows the **spatial pattern within one frequency
    band**, with its own colorbar reporting relative power for
    that band.

    The maps use independent color scales. Therefore:

    - colors can be compared spatially within one map;
    - colors should not be compared between different maps —
      always check each map's own colorbar;
    - cross-band magnitude should be interpreted using the bar
      chart above.
    """)
    return


@app.cell
def _(band_names, bandpower_df, mne, plt, raw):
    _eeg_info = raw.copy().pick("eeg").info

    _topomap_figure, _topomap_axes = plt.subplots(
        1,
        len(band_names),
        figsize=(14, 3.5),
    )

    for _ax, _band in zip(_topomap_axes, band_names):
        _values = _eeg_info["ch_names"]
        _values = bandpower_df.loc[
            _values,
            _band,
        ].values

        _im, _ = mne.viz.plot_topomap(
            _values,
            _eeg_info,
            axes=_ax,
            show=False,
        )

        # Per-band colorbar, so the relative-power units are visible
        # directly on the figure rather than only in the surrounding
        # prose. Scales are independent by design (see note above).
        plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04, label="Rel. power")

        _ax.set_title(_band)

    plt.tight_layout()
    plt.close(_topomap_figure)

    _topomap_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    ### Mathematical interpretation

    Relative band power represents the proportion of spectral
    power assigned to a frequency band relative to the total
    power across the selected 4–45 Hz range.

    Higher values therefore indicate that a larger proportion of
    the analyzed spectral power is located in that frequency band.

    ### EEG interpretation

    Band-power differences across channels describe differences
    in the spatial distribution of spectral power.

    They should not automatically be interpreted as evidence of a
    specific cognitive, neurological, or clinical state.

    In particular, the Gamma band (30–45 Hz) is more vulnerable to
    contamination from EMG, movement, electrode artifacts, and
    power-line noise. High Gamma power should therefore not be
    interpreted as cortical Gamma activity without appropriate
    artifact assessment.

    This example does not perform ICA or other artifact-removal
    procedures.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Atlas Summary

    | Field | Value |
    |---|---|
    | Library | YASA |
    | Method | Welch PSD |
    | Feature | Band Power |
    | Domain | Spectral + Spatial |
    | Input | Continuous MNE `Raw` |
    | Output | Channel × Band |
    | Bands | Theta, Alpha, Beta, Gamma |
    | Frequency range | 4–45 Hz |
    | Normalization | Relative power |
    | Absolute output | `TotalAbsPow` |
    | Visualization | Bar chart + Topomap (per-band colorbar) |
    | Main caveat | Relative power depends on selected bands |
    | EEG caveat | Gamma is susceptible to EMG and other artifacts |

    ### Feature flow

    **EEG → MNE Raw → YASA bandpower()
    → Channel × Band values → Bar / Topomap**

    This example demonstrates how a YASA feature can be extracted
    from an MNE EEG object and reused with MNE visualization tools.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    **YASA**

    Vallat, R., & Walker, M. P. (2021). An open-source,
    high-performance tool for automated sleep staging. *eLife*,
    10, e70092. https://doi.org/10.7554/eLife.70092

    Official YASA documentation and API reference:
    `yasa.bandpower` — https://yasa-sleep.org/generated/yasa.bandpower.html

    **MNE-Python / EEGBCI**

    MNE-Python documentation:
    EEGBCI dataset loading and EEGBCI run definitions —
    https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html

    **Dataset**

    EEG Motor Movement/Imagery Dataset (EEGBCI), PhysioNet —
    https://physionet.org/content/eegmmidb/1.0.0/

    **Scientific reference for EEGBCI**

    Schalk, G., McFarland, D. J., Hinterberger, T.,
    Birbaumer, N., & Wolpaw, J. R. (2004).
    BCI2000: A General-Purpose Brain-Computer Interface.
    *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.

    No local file paths are used. The EEG dataset is obtained
    through MNE's public dataset loader.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reproducibility / Environment

    Required Python packages:

    - `marimo`
    - `mne`
    - `numpy`
    - `matplotlib`
    - `yasa`

    The example uses public data and contains no hard-coded local
    dataset paths.

    Recommended project configuration:

    ```toml
    [project]
    dependencies = [
        "marimo",
        "mne",
        "numpy",
        "matplotlib",
        "yasa",
    ]
    ```

    The notebook can be launched with:

    `marimo edit <filename>.py`

    or executed as an application with:

    `marimo run <filename>.py`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final Checklist

    ✓ Clear title
    ✓ Library identified
    ✓ Method identified
    ✓ Feature identified
    ✓ Domain identified
    ✓ Dataset described
    ✓ Subject and Run specified
    ✓ Sampling frequency displayed
    ✓ Channels described
    ✓ Preprocessing visible
    ✓ No hidden transformations
    ✓ Main API call visible
    ✓ Important parameters explained
    ✓ Numerical output displayed
    ✓ Output shape explained
    ✓ Relevant visualization
    ✓ Axes and units identified (bar chart labels; per-band topomap colorbars)
    ✓ Mathematical interpretation
    ✓ EEG interpretation with appropriate caution
    ✓ No unsupported clinical claims
    ✓ Public dataset
    ✓ Official documentation identified, with links
    ✓ Scientific reference identified, with correct citation
    ✓ Atlas Summary included
    ✓ No local paths
    """)
    return


if __name__ == "__main__":
    app.run()
