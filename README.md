# EEG Feature Atlas

**EEG Feature Atlas** is an open reference project for methods, tools, and reproducible workflows for analyzing and visualizing features extracted from EEG signals.

The project is intended for researchers and students working with EEG data who need to understand not only how to compute a feature, but also how to interpret and visualize it in terms of:

* time and frequency;
* EEG channels and spatial location;
* brain regions;
* connectivity between signals or regions;
* statistical or physiological significance;
* relationships with clinical conditions, brain states, or interventions.

The repository will gradually combine a curated list of relevant software libraries, reproducible analysis pipelines, and examples based on features reported in research literature.

## Project Roadmap

The project is developed in three stages:

1. **Tools and methods**

   * collect relevant EEG and signal-analysis libraries;
   * classify available methods;
   * document what each library is useful for.

2. **Analysis pipelines**

   * build simple reproducible examples;
   * demonstrate common EEG analysis and visualization workflows;
   * compare alternative approaches where appropriate.

3. **Research feature examples**

   * reproduce and visualize selected mathematical or signal features reported in scientific literature;
   * show how a discovered feature can be localized, interpreted, and visualized;
   * include examples such as spectral features, wavelet-based features, entropy measures, and connectivity.

## Libraries

This section is the starting point of the project. It should be expanded as relevant tools are identified.

| Library        | Main Focus                           | Potential Use in EEG Feature Atlas                                                                                                             |
| -------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **MNE-Python** | EEG/MEG analysis and visualization   | EEG preprocessing, epochs, spectral analysis, time-frequency analysis, sensor-space visualization, source localization, connectivity workflows |
| **YASA**       | Sleep and EEG analysis               | Sleep staging, spectral features, sleep-related events, band-power analysis, EEG feature extraction                                            |
| **NeuroKit2**  | Neurophysiological signal processing | General biosignal processing, EEG-related signal analysis, complexity measures, multimodal physiological analysis                              |
| **AntroPy**    | Entropy and complexity analysis      | Sample entropy, permutation entropy, spectral entropy, fractal dimensions, Hjorth parameters, and other nonlinear EEG features                 |

## What to Add

When adding a new library, try to describe:

* **What problem does the library solve?**
* **Which EEG analysis methods does it implement?**
* **Which types of features can it calculate?**
* **Which visualization methods does it provide?**
* **Does it support spatial, spectral, temporal, or connectivity analysis?**
* **Is it suitable for reproducible research pipelines?**
* **What makes it different from the other libraries in this list?**

Useful categories may include:

* preprocessing;
* artifact removal;
* temporal features;
* spectral features;
* time-frequency analysis;
* wavelets;
* entropy and complexity;
* spatial analysis;
* source localization;
* connectivity;
* graph-based analysis;
* statistical analysis;
* machine-learning features;
* visualization.

## Project Status

The repository is currently in the **tools and methods discovery stage**.

Initial work focuses on building a structured overview of libraries and methods that can later be used to create reproducible EEG analysis and visualization pipelines.
