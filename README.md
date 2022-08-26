# Simulating the central tendency effect in different social contexts using a predictive coding model

Code for replicating the results of the paper "The world seems different in a social context: a neural network analysis of human experimental data" authored by Maria Tsfasman, Anja Philippsen, Carlo Mazzola, Serge Thill, Alessandra Sciutti, and Yukie Nagai.

Link to the arXiv version: https://arxiv.org/pdf/2203.01862.pdf

## Required python packages
* chainer
* numpy
* matplotlib
* seaborn
* dtw
* scikit-learn

## Steps for reproducing results

1. Run `run_training.py` to train the network with the human experiment data. This step can be skipped if the pretrained models provided in `results/training/all_human/` should be used.

2. Run `evaluate_training.py` to evaluate the networks' performance and run the internal representation analysis (the paths for the network which should be evaluated needs to be adjusted). The evaluation is done automatically with a number of different H (prior/sensory integration) values. The results are written to a subfolder `evaluation` inside the experimental results folder. If results are already available in this folder, they are reused.

3. Run `modify_H_behavior.py` to test which change of the precision of the prior or on sensory information

4. Run `evaluate_training_summary_across_network.py` to summarize the network results and compute which H values best replicate human data and to compute the distances between neuron activations of different conditions (see Figures 8 and 9 of the manuscript).

5. Run `modify_H_behavior_statistics.py` which processes the results of `modify_H_behavior.py` and saves it in a csv format to be used for statistical analysis with R.

6. Run `statistics.R` with R to test statistical difference between the three conditions of the human and of the model results.

## Description of source code files

### `nets.py`
SCTRNN implementation used the the network training code and to load stored networks into the correct data structure.

### `run_training.py`
Trains the network with human data.

### `evaluate_training.py`
Evaluates the network performance of all trained networks, including internal representation analysis.

### `evaluate_training_summary_across_networks.py`
Summarizes the results obtained from several networks (requires results of `evaluate_training.py` and of `modify_H_behavior.py`).

### `modify_H_behavior.py`
Changing the precision on prior (by H) and on sensory information (by sigma_inp) to a number of different values, and compare it to human data.
Compares the effect of the change of either prior or sensory information (both affect the result in similar ways).

### `network_activation_summary.py`
Computes the distances within conditions (see Figures 8 and 9 of the manuscript).

### `utils`
Contains helper functions, e.g. `quantitative_distances.py` contains helper functions to compute distances between time series (behavioral trajectory, activation history, or PCA activations), and to sort the full data results by either length and condition, or by participant and condition. `calc_regression_indices.py` calculates the regression index.

### `statistics.R`
Analyses of statistical significance between human and model results (`human-model-comparison.csv`) using linear mixed effect models and likelihood-ratio tests.

## Description of other files

### Data

The human data, preprocessed for usage with the neural network model is located in the folder `human_data`.

* `presented.npy` and `presented_norm.npy` contain the presented stimuli (original and normalized).

* `human.npy` and `human_norm.npy` contain the human reproductions of the stimuli (original and normalized).

* `subjects.npy` maps the data entries to the human subject ID (0-24).

* `conditions.npy` maps the data entries to the experimental conditions: 0 individual, 1 mechanical and 2 social condition.

* `classes.npy` maps the data entries to the initial states that the network uses for training. Each subject and each condition are represented by a different initial state, thus, 75 initial states are used in total.

### Results

The ten trained networks used for the analysis are stored in
`results/training/all_human/` which contains a subfolder with the name of the training `H_prior` parameter (here, always `H_prior = 1`). Inside, the trained network is stored as `network-final.npz`.

The training parameters and the trained network can be loaded using the nets.load_network() function:

`params, trained_model = load_network(PATH_TO_FOLDER, model_filename='network-final.npz')`

In the `evaluation` subfolder, all evaluation results are stored.

