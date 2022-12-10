# LegNet: solving the sequence-to-expression problem with SOTA convolutional networks

Here we present a convolutional network for predicting gene expression and sequence variant effects based on data obtained by large-scale parallel reporter assays. 

Our approach secured 1st place in the recent [DREAM 2022 challenge in predicting gene expression from millions of promoter sequences](https://www.synapse.org/#!Synapse:syn28469146/wiki/619131). To achieve the top performance, we drew inspiration from EfficientNetV2, a recent state-of-the-art in image analysis, and rephrased the initial sequence-to-expression regression problem as a soft-classification task. In the framework of the DREAM challenge, our model outperformed both attention transformers and recurrent neural networks.

## This repository provides several resources:

- A [tutorial Jupyter notebook](tutorial/demo_notebook.ipynb) demonstrating how LegNet can be practically used with the data from yeast gigantic parallel reporter assays.

- [Scripts](scripts/paper) to reproduce the analysis presented in the LegNet manuscript based on the public GPRA data of [Vaishnav et al.](https://doi.org/10.1038/s41586-022-04506-6), [Zenodo](https://zenodo.org/record/4436477#.Y5R6IOxBy3J).

- [Scripts](scripts/dream2022) to reproduce the autosome.org solution for the [DREAM 2022 promoter expression challenge](https://www.synapse.org/#!Synapse:syn28469146/wiki/619131).
