import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import antropy as ant
    import mne

    from mne.channels import make_standard_montage
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    return (
        ant,
        eegbci,
        make_standard_montage,
        mne,
        mo,
        np,
        plt,
        read_raw_edf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AntroPy: Spectral Entropy on Real EEG

    This example demonstrates how **spectral entropy** — computed
    with **AntroPy (`spectral_entropy`)** — behaves on real EEG
    recordings rather than on synthetic signals.

    The workflow follows the EEG Feature Atlas structure:

    **Signal → Method → Feature → Numerical Output
    → Visualization → Interpretation**

    We will:

    1. download three real EEG recordings for one subject (rest,
       motor execution, motor imagery);
    2. compare spectral entropy for one channel across these three
       states;
    3. compare spectral entropy across several channels for one
       state;
    4. compare spectral entropy across frequency bands for one
       channel and one state;
    5. visualize the spatial distribution of spectral entropy
       across the scalp, per frequency band;
    6. interpret the feature carefully;
    7. summarize the feature for the Atlas.

    The goal of these comparisons is **not** to prove a
    physiological law or build a classifier — it is simply to show
    clearly that spectral entropy is a real, measurable property of
    EEG signals that varies with the state, the channel, and the
    frequency band chosen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Library / Method / Feature

    | Field | Value |
    |---|---|
    | Library | AntroPy |
    | Function | `ant.spectral_entropy()` |
    | Method | Shannon entropy of the power spectral density (PSD) |
    | Feature | Spectral entropy (complexity measure) |
    | Domain | Spectral / Signal complexity |
    | Input | 1-D signal (single EEG channel, optionally band-limited) |
    | Output | Single scalar value per input signal |
    | Visualization | Bar charts (cross-condition comparison) + scalp topomaps |

    **Important distinction:**

    - PSD estimation (via `method='welch'`) = spectral estimation
      step, internal to the function;
    - Spectral entropy = the feature extracted from that PSD;
    - Bar charts / topomaps = visualizations used to compare the
      resulting scalar values across conditions, channels, and
      scalp locations.

    Spectral entropy is defined as the Shannon entropy of the
    normalized PSD:

    **H(x, sf) = −Σ P(f) · log₂[P(f)]**

    where P is the normalized PSD and the sum runs over all
    frequencies from 0 to sf/2. When `normalize=True`, this value
    is divided by log₂(N), where N is the number of frequency
    bins, so the result is bounded to **[0, 1]** regardless of
    signal length or sampling rate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature parameters

    The important parameters are specified explicitly rather than
    relying on silently-applied defaults:

    - `method="welch"` – estimate the PSD with Welch's method
      rather than a single periodogram (`'fft'`); Welch averages
      over multiple overlapping segments, giving a smoother PSD
      estimate at the cost of frequency resolution.
    - `normalize=True` – divide the raw entropy (in bits) by
      log₂(N) so the result is bounded to [0, 1] and comparable
      across signals of different length or sampling rate.
    - `nperseg` (not set here) – AntroPy leaves this at `None`,
      which defers to `scipy.signal.welch`'s own default of
      **256 samples per segment**. At the EEGBCI sampling rate
      (`sf=160 Hz`) this corresponds to ~1.6 s segments. Changing
      `nperseg` changes frequency resolution and can change the
      resulting entropy value, so it should be reported when
      comparing results across studies.

    AntroPy's `spectral_entropy` has no built-in frequency-band
    limit, so the **band-specific comparisons below band-pass
    filter the EEG channel first** (via MNE's `Raw.filter()`) and
    then compute spectral entropy on the filtered signal. This
    filtering step is a real, visible transformation of the input
    — not a hidden default — and it directly determines which part
    of the spectrum the entropy value describes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset

    We use the public **EEG Motor Movement/Imagery Dataset
    (EEGBCI)** distributed through PhysioNet — the same dataset
    used in the MNE and YASA examples in this Atlas.

    For one subject, we compare three recordings that correspond
    to three different states, using the run mapping documented for
    EEGBCI:

    | State | Run | Description |
    |---|---|---|
    | Rest | 2 | Baseline, eyes closed |
    | Execution | 3 | Task 1 — real (executed) opening/closing of left or right fist |
    | Imagery | 4 | Task 2 — imagined opening/closing of left or right fist |

    The data are downloaded automatically by MNE if they are not
    already available locally.
    """)
    return


@app.cell
def _(eegbci, make_standard_montage, read_raw_edf):
    def load_run(subject, run):
        """Download (if needed), load, and prepare one EEGBCI run."""
        eeg_files = eegbci.load_data(subject, [run])

        raw = read_raw_edf(
            eeg_files[0],
            preload=True,
            verbose=False,
        )

        # Preprocessing: standardize EEGBCI channel names to MNE-compatible
        # names, then attach standard 10-05 electrode positions so the
        # recording can be used for the scalp topomaps below. No filtering,
        # resampling, epoching, ICA, or artifact removal is applied.
        eegbci.standardize(raw)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        return raw

    return (load_run,)


@app.cell
def _(load_run):
    subject = 1
    state_runs = {"Rest": 2, "Execution": 3, "Imagery": 4}

    raws = {state: load_run(subject, run) for state, run in state_runs.items()}
    return raws, state_runs, subject


@app.cell(hide_code=True)
def _(mo, raws, state_runs, subject):
    _rows = "\n".join(
        f"| {state} | {state_runs[state]} | {raws[state].info['sfreq']:.1f} Hz | "
        f"{len(raws[state].ch_names)} | {raws[state].duration:.1f} s |"
        for state in raws
    )

    mo.md(f"""
    ## Recording information

    | State | Run | Sampling frequency | Channels | Duration |
    |---|---|---|---|---|
    {_rows}

    Subject: **{subject}**. Only channel-name standardization and
    montage attachment are applied to each recording — no
    filtering, resampling, epoching, ICA, or artifact removal
    beyond what each comparison below applies explicitly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison 1 — One channel, across states

    Using a single central channel (`Cz`), we compute the
    (full-band, unfiltered) normalized spectral entropy for three
    states.

    The **rest** recording (run 2) is a single continuous baseline,
    so the full run is used. The **execution** and **imagery**
    recordings (runs 3 and 4) are not pure task recordings — each
    is annotated with alternating `T0` (rest), `T1` (left-fist
    onset), and `T2` (right-fist onset) segments, with `T0` making
    up roughly half of the run. Computing entropy over the full run
    would mix task and rest together and dilute the very contrast
    we want to show, so for these two states we concatenate only
    the `T1` (left-fist) task epochs from the annotations before
    computing spectral entropy — keeping the movement side fixed
    so laterality doesn't become a second, uncontrolled variable
    alongside state.

    All three values still use the same channel and the same
    `spectral_entropy` parameters, so they remain directly
    comparable.
    """)
    return


@app.cell
def _(ant, np, raws):
    def extract_task_epochs(raw, channel, task_code="T1"):
        """Concatenate only the annotated `task_code` segments for one channel.

        Runs 3 and 4 interleave rest (T0) with task trials (T1/T2), so
        slicing out just the T1 (left-fist) onsets gives a signal that
        actually reflects the task, rather than ~50% rest + ~50% task.
        """
        sf = raw.info["sfreq"]
        segments = [
            raw.get_data(
                picks=[channel],
                start=int(round(onset * sf)),
                stop=int(round((onset + duration) * sf)),
            )[0]
            for onset, duration, description in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description,
            )
            if description == task_code
        ]
        return np.concatenate(segments)

    state_comparison_channel = "Cz"

    state_entropies = {}
    for _state, _raw in raws.items():
        _sf = _raw.info["sfreq"]
        if _state == "Rest":
            _sig = _raw.get_data(picks=[state_comparison_channel])[0]
        else:
            _sig = extract_task_epochs(_raw, state_comparison_channel, task_code="T1")
        state_entropies[_state] = ant.spectral_entropy(
            _sig, sf=_sf, method="welch", normalize=True
        )

    return state_comparison_channel, state_entropies


@app.cell
def _(plt, state_comparison_channel, state_entropies):
    _fig, _ax = plt.subplots(figsize=(6, 4))

    _ax.bar(
        state_entropies.keys(),
        state_entropies.values(),
        color=["#55A868", "#DD8452", "#4C72B0"],
    )
    _ax.set_ylabel("Normalized spectral entropy")
    _ax.set_ylim(0, 1)
    _ax.set_title(f"Spectral Entropy Across States (channel {state_comparison_channel})")
    _ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo, state_comparison_channel, state_entropies):
    _lines = "\n".join(
        f"- **{state}:** {value:.3f}" for state, value in state_entropies.items()
    )
    mo.md(f"""
    **Channel `{state_comparison_channel}`, normalized spectral entropy:**

    {_lines}

    These three values were computed with identical parameters
    (`method="welch"`, `normalize=True`, same channel). Rest uses
    the full baseline run; execution and imagery use only their
    concatenated `T1` (left-fist) task epochs, so all three
    reflect the intended state rather than a mix of state and rest.
    Any difference between them reflects the recording itself, not
    the analysis settings. This shows that spectral entropy is not
    a fixed property of a channel — it changes with the state of
    the subject during recording.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison 2 — One state, across channels

    Using the imagery recording, we compute spectral entropy for
    three channels spanning frontal, central, and
    parietal scalp regions.
    """)
    return


@app.cell
def _(ant, raws):
    channel_comparison_state = "Imagery"
    comparison_channels = ["Fz", "Cz", "Pz"]

    _raw = raws[channel_comparison_state]
    _sf = _raw.info["sfreq"]

    channel_entropies = {
        ch: ant.spectral_entropy(
            _raw.get_data(picks=[ch])[0], sf=_sf, method="welch", normalize=True
        )
        for ch in comparison_channels
    }

    return channel_comparison_state, channel_entropies, comparison_channels


@app.cell
def _(channel_comparison_state, channel_entropies, plt):
    _fig, _ax = plt.subplots(figsize=(6, 4))

    _ax.bar(
        channel_entropies.keys(),
        channel_entropies.values(),
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    _ax.set_ylabel("Normalized spectral entropy")
    _ax.set_ylim(0, 1)
    _ax.set_title(f"Spectral Entropy Across Channels ({channel_comparison_state} state)")
    _ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(channel_comparison_state, channel_entropies, mo):
    _lines = "\n".join(
        f"- **{ch}** (Fz = frontal, Cz = central, Pz = parietal): {value:.3f}"
        for ch, value in channel_entropies.items()
    )
    mo.md(f"""
    **{channel_comparison_state} state, normalized spectral entropy:**

    {_lines}

    All three values come from the same recording and the same
    time window, computed with identical parameters — only the
    channel (scalp location) differs. This shows that spectral
    entropy is not uniform across the scalp at a given moment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison 3 — One channel and state, across frequency bands

    Using channel `Cz` in the imagery recording, we band-pass
    filter the signal into three classic EEG bands before computing
    spectral entropy on each filtered version. This measures how
    concentrated or spread out the spectrum is *within* each band,
    rather than across the full spectrum.
    """)
    return


@app.cell
def _(ant, raws):
    band_comparison_state = "Imagery"
    band_comparison_channel = "Cz"
    comparison_bands = {"Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

    _raw = raws[band_comparison_state]
    _sf = _raw.info["sfreq"]

    band_entropies = {}
    for _band_name, (_lo, _hi) in comparison_bands.items():
        _raw_band = _raw.copy().filter(
            l_freq=_lo, h_freq=_hi, picks=[band_comparison_channel], verbose=False
        )
        _sig = _raw_band.get_data(picks=[band_comparison_channel])[0]
        band_entropies[_band_name] = ant.spectral_entropy(
            _sig, sf=_sf, method="welch", normalize=True
        )

    return band_comparison_channel, band_comparison_state, band_entropies, comparison_bands


@app.cell
def _(band_comparison_channel, band_comparison_state, band_entropies, plt):
    _fig, _ax = plt.subplots(figsize=(6, 4))

    _ax.bar(
        band_entropies.keys(),
        band_entropies.values(),
        color=["#8172B2", "#CCB974", "#64B5CD"],
    )
    _ax.set_ylabel("Normalized spectral entropy")
    _ax.set_ylim(0, 1)
    _ax.set_title(
        f"Spectral Entropy Across Bands "
        f"(channel {band_comparison_channel}, {band_comparison_state} state)"
    )
    _ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(
    band_comparison_channel,
    band_comparison_state,
    band_entropies,
    comparison_bands,
    mo,
):
    _lines = "\n".join(
        f"- **{band}** ({comparison_bands[band][0]}–{comparison_bands[band][1]} Hz): {value:.3f}"
        for band, value in band_entropies.items()
    )
    mo.md(f"""
    **Channel `{band_comparison_channel}`, {band_comparison_state} state,
    normalized spectral entropy per band:**

    {_lines}

    Each value is computed on the same channel and recording, after
    band-pass filtering to a different frequency range. This shows
    that spectral entropy is not a single number for a channel —
    it depends on which part of the spectrum is included, since a
    narrower band can concentrate or spread out power differently
    than the full spectrum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatial distribution of spectral entropy (topomap)

    Beyond single-channel comparisons, spectral entropy can be
    computed independently for every EEG channel and displayed as a
    scalp topomap — the same visualization already used in this
    Atlas for spectral **power** (`mne/01_spectrum.py`).

    The key difference: the power topomap shows how much energy
    each region has in a band. This topomap instead shows, within
    each band, how *concentrated or spread out* that region's
    spectrum is — a channel can have strong power in a band while
    still having high or low spectral entropy in that same band.

    We use the rest recording (run 2) — the same recording used for
    the power topomap in `mne/01_spectrum.py` — so the two figures
    are directly comparable.
    """)
    return


@app.cell
def _(ant, mne, np, raws):
    topomap_state = "Rest"
    topomap_bands = {"Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}

    _raw = raws[topomap_state]
    _picks = mne.pick_types(_raw.info, eeg=True)
    _sf = _raw.info["sfreq"]

    topomap_entropies = {}
    for _band_name, (_lo, _hi) in topomap_bands.items():
        _raw_band = _raw.copy().filter(l_freq=_lo, h_freq=_hi, picks=_picks, verbose=False)
        _data = _raw_band.get_data(picks=_picks)
        topomap_entropies[_band_name] = np.array([
            ant.spectral_entropy(_data[i], sf=_sf, method="welch", normalize=True)
            for i in range(_data.shape[0])
        ])

    topomap_picks = _picks
    return topomap_bands, topomap_entropies, topomap_picks, topomap_state


@app.cell
def _(mne, plt, raws, topomap_bands, topomap_entropies, topomap_picks, topomap_state):
    _raw = raws[topomap_state]

    _fig, _axes = plt.subplots(1, 4, figsize=(15, 4))

    _picked_info = mne.pick_info(_raw.info, topomap_picks)

    for _ax, _band_name in zip(_axes, topomap_bands):
        _values = topomap_entropies[_band_name]

        _im, _cn = mne.viz.plot_topomap(
            _values,
            _picked_info,
            axes=_ax,
            show=False,
            cmap="Reds",
        )
        _ax.set_title(_band_name)
        _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04, label="Entropy")

    _fig.suptitle(f"Spatial Distribution of Spectral Entropy ({topomap_state} state)")
    plt.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    ### Mathematical interpretation

    Spectral entropy is the (normalized) Shannon entropy of the
    signal's power spectral density. A value near 0 means nearly
    all spectral power sits in a narrow frequency range; a value
    near 1 means power is close to uniformly distributed across all
    resolved frequencies (or, for the band-limited and topomap
    figures above, across the frequencies within that band). It
    says nothing about *which* frequencies carry the power, only
    how concentrated or spread out the distribution is.

    ### EEG interpretation

    The comparisons above show that spectral entropy varies with
    three things that must always be reported alongside a value:
    the **state** the subject was in during recording, the
    **channel** (scalp location), and the **frequency band**
    considered. A single spectral entropy value, without this
    context, is not by itself evidence of a specific cognitive,
    neurological, or clinical state.

    It also depends on parameters such as `method` and `nperseg`
    (see Feature parameters above) — values are only directly
    comparable when computed with matching parameters. This example
    does not apply artifact removal to the EEG channels, so
    broadband noise or artifacts could inflate the measured
    entropy; a full pipeline would typically clean the signal
    before computing this feature.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Atlas Summary

    | Field | Value |
    |---|---|
    | Library | AntroPy |
    | Method | Shannon entropy of Welch PSD |
    | Feature | Spectral entropy |
    | Domain | Spectral / Complexity |
    | Input | 1-D signal, single EEG channel, optionally band-pass filtered |
    | Output | Scalar per channel/band, bounded to [0, 1] when `normalize=True` |
    | Visualization | Bar charts (state / channel / band comparisons) + scalp topomaps |
    | Main caveat | Value depends on `method` and `nperseg`; not directly comparable across mismatched parameters |
    | EEG caveat | No artifact removal applied here; broadband noise can inflate the value |

    ### Feature flow

    **EEG channel (optionally band-filtered) → AntroPy
    spectral_entropy() → scalar value → bar chart comparison /
    per-channel scalp topomap**

    This example demonstrates that the same AntroPy feature, with
    the same parameters, produces different values depending on
    the recording state, the channel, and the frequency band —
    and that computing it per channel yields a spatial map
    comparable to (but distinct from) a spectral power topomap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    **AntroPy**

    Vallat, R. AntroPy: entropy and complexity of (EEG)
    time-series in Python. Official documentation and API
    reference: `spectral_entropy` —
    https://raphaelvallat.com/antropy/

    **Scientific reference for spectral entropy**

    Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S.,
    Iyama, A., Katsuda, Y., & Hirano, M. (1991). Quantification of
    EEG irregularity by use of the entropy of the power spectrum.
    *Electroencephalography and Clinical Neurophysiology*, 79(3),
    204–210.

    **MNE-Python / EEGBCI**

    MNE-Python documentation: EEGBCI dataset loading and run
    definitions —
    https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html

    **Dataset**

    EEG Motor Movement/Imagery Dataset (EEGBCI), PhysioNet —
    https://physionet.org/content/eegmmidb/1.0.0/

    **Scientific reference for EEGBCI**

    Schalk, G., McFarland, D. J., Hinterberger, T.,
    Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: A
    General-Purpose Brain-Computer Interface. *IEEE Transactions
    on Biomedical Engineering*, 51(6), 1034–1043.

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
    - `antropy`

    The example uses only public data (EEGBCI, subject 1, runs 2/3/4)
    and contains no hard-coded local dataset paths.

    Recommended project configuration:

    ```toml
    [project]
    dependencies = [
        "marimo",
        "mne",
        "numpy",
        "matplotlib",
        "antropy",
    ]
    ```

    The notebook can be launched with:

    `uv run marimo edit <filename>.py`

    or executed as an application with:

    `uv run marimo run <filename>.py`
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
    ✓ Subject and runs specified
    ✓ Sampling frequency displayed
    ✓ Channels described
    ✓ Preprocessing visible
    ✓ No hidden transformations (including `nperseg` default and
      band-pass filtering for band-limited comparisons)
    ✓ Main API call visible
    ✓ Important parameters explained
    ✓ Numerical output displayed
    ✓ Output shape explained (scalar, [0, 1])
    ✓ Relevant visualization (bar charts + scalp topomaps)
    ✓ Axes and units identified
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
