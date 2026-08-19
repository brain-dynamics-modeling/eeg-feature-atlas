import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Spectral Entropy with AntroPy

    This example demonstrates how to calculate **spectral entropy**
    from a signal using AntroPy.

    Spectral entropy measures how the signal's power is distributed
    across frequencies.

    - Lower spectral entropy → power is concentrated in a small number of frequencies.
    - Higher spectral entropy → power is spread across many frequencies.

    We will first compare a simple periodic signal with a noisy signal
    to understand the behavior of the feature.
    """)
    return


@app.cell
def _():
    import antropy as ant
    import matplotlib.pyplot as plt
    import numpy as np

    return ant, np, plt


@app.cell
def _(mo):
    mo.md("""
    ## Create a Simple 10 Hz Signal

    We first create a simple periodic signal at **10 Hz**.

    A 10 Hz rhythm is within the **alpha frequency range**
    commonly studied in EEG.

    This signal is intentionally simple and regular,
    so its power is concentrated around one frequency.
    """)
    return


@app.cell
def _(np):
    sf = 100
    duration = 10

    time = np.arange(0, duration, 1 / sf)

    clean_signal = np.sin(2 * np.pi * 10 * time)
    return clean_signal, sf, time


@app.cell
def _(mo):
    mo.md("""
    ## Visualize the Signal

    Before calculating spectral entropy, we visualize the generated signal.

    Because this is a clean 10 Hz sine wave, we expect to see
    a regular and repeating pattern.
    """)
    return


@app.cell
def _(clean_signal, plt, time):
    _fig, _ax = plt.subplots()

    _ax.plot(time, clean_signal)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Amplitude")
    _ax.set_title("10 Hz Sine Wave")

    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Create a Noisy Signal

    Next, we create a random noisy signal.

    Unlike the 10 Hz sine wave, the noisy signal contains
    a broader range of frequency components.

    We will later compare the spectral entropy of the regular signal
    with the spectral entropy of the noisy signal.
    """)
    return


@app.cell
def _(np, time):
    rng = np.random.default_rng(42)

    noisy_signal = rng.normal(size=time.shape)
    return (noisy_signal,)


@app.cell
def _(mo):
    mo.md("""
    ## Visualize the Noisy Signal

    We now visualize the random noisy signal.

    Unlike the regular 10 Hz sine wave, this signal changes
    irregularly over time and does not show a clear repeating pattern.
    """)
    return


@app.cell
def _(noisy_signal, plt, time):
    fig, ax = plt.subplots()

    ax.plot(time, noisy_signal)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Random Noisy Signal")

    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Calculate Spectral Entropy

    We now calculate the normalized spectral entropy of both signals
    using AntroPy.

    The 10 Hz sine wave is expected to have lower spectral entropy
    because most of its power is concentrated around one frequency.

    The noisy signal is expected to have higher spectral entropy
    because its power is spread across a wider range of frequencies.
    """)
    return


@app.cell
def _(ant, clean_signal, noisy_signal, sf):
    clean_entropy = ant.spectral_entropy(
        clean_signal,
        sf=sf,
        method="welch",
        normalize=True,
    )

    noisy_entropy = ant.spectral_entropy(
        noisy_signal,
        sf=sf,
        method="welch",
        normalize=True,
    )
    return clean_entropy, noisy_entropy


@app.cell
def _(mo):
    mo.md("""
    ## Compare the Results

    We compare the normalized spectral entropy of the two signals.

    We expect:

    - the clean 10 Hz sine wave to have lower spectral entropy;
    - the noisy signal to have higher spectral entropy.
    """)
    return


@app.cell
def _(clean_entropy, noisy_entropy):
    print(f"Clean 10 Hz signal: {clean_entropy:.3f}")
    print(f"Noisy signal: {noisy_entropy:.3f}")
    return


if __name__ == "__main__":
    app.run()
