import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import mne
    import numpy as np

    from mne.channels import make_standard_montage
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    return eegbci, make_standard_montage, mo, np, plt, read_raw_edf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MNE-Python: EEG Power Spectrum

    This example demonstrates basic frequency-domain analysis of EEG data
    using the `Spectrum` class in MNE-Python.

    We will:

    1. download a small public EEG recording;
    2. load it as an MNE `Raw` object;
    3. attach standard EEG sensor positions;
    4. calculate the power spectral density (PSD);
    5. visualize the spectrum;
    6. visualize the spatial distribution of spectral power.

    The example is inspired by the official MNE tutorial:

    **The Spectrum and EpochsSpectrum classes: frequency-domain data**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset

    We use the EEG Motor Movement/Imagery Dataset distributed through
    PhysioNet and directly accessible through MNE.

    The dataset contains 64-channel EEG recordings.

    For this example we use:

    - **Subject:** 1
    - **Run:** 2
    - **Condition:** resting-state EEG

    Only one recording is downloaded.
    """)
    return


@app.cell
def _(eegbci):
    subject = 1
    runs = [2]

    eeg_files = eegbci.load_data(subject, runs)

    eeg_files
    return (eeg_files,)


@app.cell
def _(eeg_files, eegbci, make_standard_montage, read_raw_edf):
    raw = read_raw_edf(
        eeg_files[0],
        preload=True,
        verbose=False,
    )

    # Convert EEGBCI channel names to standard MNE names.
    eegbci.standardize(raw)

    # EEGBCI follows the international 10-10 electrode system.
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    raw
    return (raw,)


@app.cell(hide_code=True)
def _(mo, raw):
    mo.md(f"""
    ## Recording information

    The recording contains:

    - **Channels:** {len(raw.ch_names)}
    - **Sampling frequency:** {raw.info["sfreq"]:.1f} Hz
    - **Duration:** {raw.duration:.1f} seconds

    The data are represented by an MNE `Raw` object.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Power Spectral Density

    Power spectral density describes how signal power is distributed
    across frequencies.

    MNE provides `Raw.compute_psd()` for calculating a spectral
    representation directly from continuous EEG data.

    Here we calculate frequencies between **1 and 45 Hz** using
    Welch's method.
    """)
    return


@app.cell
def _(raw):
    spectrum = raw.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=45.0,
        picks="eeg",
        n_fft=2048,
        verbose=False,
    )

    spectrum
    return (spectrum,)


@app.cell
def _(spectrum):
    psd_data, frequencies = spectrum.get_data(return_freqs=True)

    (
        psd_data.shape,
        frequencies.shape,
    )
    return frequencies, psd_data


@app.cell(hide_code=True)
def _(frequencies, mo, psd_data):
    mo.md(f"""
    The resulting spectral representation contains:

    - **EEG channels:** {psd_data.shape[0]}
    - **Frequency bins:** {psd_data.shape[1]}
    - **Frequency range:** {frequencies[0]:.2f}–{frequencies[-1]:.2f} Hz

    Each value represents estimated spectral power for one EEG channel
    at one frequency.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average EEG spectrum

    MNE's `Spectrum.plot()` displays spectral power as a function of
    frequency.

    The plot allows us to inspect the overall frequency structure of
    the EEG recording.
    """)
    return


@app.cell
def _(plt, spectrum):
    spectrum_figure = spectrum.plot(
        picks="eeg",
        average=True,
        dB=True,
        show=False,
    )

    plt.close(spectrum_figure)

    spectrum_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatial distribution of spectral power

    EEG frequency content can also be represented spatially.

    `Spectrum.plot_topomap()` aggregates spectral power inside
    predefined frequency bands and interpolates the values across
    the scalp.

    We display four common EEG bands:

    - **Theta:** 4–8 Hz
    - **Alpha:** 8–12 Hz
    - **Beta:** 12–30 Hz
    - **Gamma:** 30–45 Hz
    """)
    return


@app.cell
def _(np, plt, spectrum):
    bands = {
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 45),
    }

    topomap_figure = spectrum.plot_topomap(
        bands=bands,
        ch_type="eeg",
        agg_fun=np.mean,
        dB=True,
        show=False,
    )

    plt.close(topomap_figure)

    topomap_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this example demonstrates

    MNE can represent a continuous EEG recording in the frequency
    domain using a `Spectrum` object.

    From the same object we can obtain:

    - numerical PSD values;
    - frequency bins;
    - an averaged frequency × power visualization;
    - spatial scalp maps for selected frequency bands.

    This establishes a basic pattern that will be reused in later
    EEG Feature Atlas examples:

    **signal → mathematical feature → spatial / spectral visualization**

    Future examples can extend this workflow to individual channels,
    epochs, experimental conditions, wavelets, connectivity measures,
    and features reported in scientific studies.
    """)
    return


if __name__ == "__main__":
    app.run()
