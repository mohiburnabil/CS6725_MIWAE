# CS6725: Missing Data Imputation with MIWAE

This project uses the **Missing-data Importance-Weighted Autoencoder (MIWAE)** to impute 26 numerical, categorical and mixed datasets. Two course notebooks run the supplied missingness variants and produce the grading workbook. The original PyTorch tutorial demonstrates continuous imputation on Iris and compares conventional baselines.

The implementation follows the MIWAE tutorial described in the notebook and the paper by Pierre-Alexandre Mattei and Jes Frellsen, *MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets* (ICML 2019).

## Course benchmark

Run [MIWAE_Numerical.ipynb](MIWAE_Numerical.ipynb) for the 16 numerical datasets and [MIWAE_Categorical.ipynb](MIWAE_Categorical.ipynb) for the five categorical and five mixed datasets. Each notebook contains its CPU training and reporting code. The **Forward pass** cell defines the model, the **MIWAE loss** section defines the observed-data objective, and the **Training loop** cell calls that loss for each minibatch. Both notebooks import plotting functions from [miwae_figures.py](miwae_figures.py). Install the dependencies using the setup instructions below. The original ProbAI notebook is a separate Iris tutorial.

| Course input | Role |
| --- | --- |
| Original Datasets Without Labels | Ground truth for evaluation, one headerless workbook per dataset |
| Incomplete Datasets Without Labels | 44 headerless workbooks per dataset, each used for one independent fit |
| List of Datasets.xlsx | Dataset inventory and nominal feature counts |
| Instructions.pdf | Course evaluation formulas and delivery requirements |
| Table-NRMS-AE.xlsx | Blank reporting template, one row per variant plus averages and complexity |

The 44 combinations are the file codes C, AE, AG, AL, AN, AW, NE, NG, NL, NN and NW at nominal rates of 1, 5, 10 and 20 percent. Actual masks come from blank cells in each supplied workbook. The benchmark is transductive. Training sees the observed entries of the same rows whose missing entries are scored.

### Select data and rerun

In either notebook, leave `DATASET = None` to process its group in sequence, or choose one name from the printed list. For example, `DATASET = "Adult"` and `VARIANT = "C_10"` run one mixed-data experiment. Leaving `VARIANT = None` runs all 44 masks for the selected dataset. Set `GROUP = "all"` to access all 26 datasets from either notebook.

By default the notebooks reuse completed variants with matching settings. To train afresh, change `OUTPUT_DIR` to a new directory. Interruptions preserve completed variants. Both notebooks rebuild the workbook from the results in the same output folder. To run the whole benchmark, set `GROUP = "all"`, `DATASET = None`, and `VARIANT = None` in either notebook and run all cells from the project directory.

Each run saves a completed table as Parquet and a JSON record of its scores, settings, epoch count and elapsed time under `outputs/miwae/imputed/<dataset>/`. Read a table with `pandas.read_parquet`. Columns x1, x2, … follow the input workbook order. The [full benchmark workbook](outputs/miwae_full/Table-NRMS-AE.xlsx) retains the template’s dataset names.

### Browse report figures

Open the [full benchmark gallery](outputs/miwae_full/visualizations/index.html) in a browser, select a dataset, then choose a mask code and missingness rate. Each image links to a standalone PNG that can be inserted into a report. The [current run gallery](outputs/miwae/visualizations/index.html) follows the notebook’s output selection.

Both notebooks create the same hierarchy under the selected output folder:

```text
visualizations/
  index.html
  common/                 Across-dataset comparisons
  Iris/                   Comparisons among completed Iris versions
    Iris_C_10/            Diagnostics for one imputation
```

| Question | Figures |
| --- | --- |
| Where is data missing? | Row-by-feature mask, percentage missing per feature, distribution of missing features per row |
| How well are numerical values recovered? | Feature NRMS, true-versus-imputed scatter or hexbin, overlaid histograms, signed residuals, absolute-error ECDF |
| Which categories are imputed incorrectly? | Feature AE, row-normalized confusion matrix, true and imputed category frequencies |
| How did optimization progress? | Sampled minibatch MIWAE loss by optimizer update for fresh fits, plus verified replays for the C_10 version of every dataset in the full gallery |
| How do versions of one dataset compare? | Mask-by-rate score heatmaps, rate trajectories, boxplots with individual version scores, mean-score bars by mask code, runtime by rate |
| How do datasets compare? | Score boxplots, dataset-by-rate and dataset-by-mask heatmaps, paired low/high-rate means, median rate trends with interquartile bands, score ECDFs, mixed-data NRMS versus AE, epoch counts, runtime and score coverage |

Feature summaries use all applicable cells. The mask image shows up to 250 reproducibly sampled rows in source order. Detailed recovery figures select the feature with the most hidden cells of each type. Numerical panels use the same reproducible sample of up to 5,000 hidden pairs. Categorical panels use every hidden pair of the selected feature and pool labels beyond the nine most frequent. A figure is omitted when its metric has no eligible cells. Aggregate plots use the completed records available in that output folder, with equal weight per scored version. Elapsed time excludes figure rendering and depends on machine load.

The C_10 training curves in the full gallery come from deterministic replays. Each replay reproduced every saved imputed value exactly before its loss trace was added. Other historical versions retain their original score and recovery diagnostics.

Figures are enabled by default in `run()`. Rerunning a completed selection backfills missing version images from its saved imputation. To refresh only the comparison galleries after moving or removing outputs, call `comparison_galleries(OUTPUT_DIR)` from `miwae_figures`.

### Feature types and scores

Numerical features use a Student’s t likelihood and standardization fitted to observed entries. The encoder receives zero-filled numerical values, one-hot categorical inputs and observed indicators. Each categorical feature has its own softmax likelihood. Mixed models learn both types jointly under one latent vector. Imputation uses importance-weighted conditional means for numbers and the class with maximum importance-weighted probability for categories. The runner fills only missing cells and checks observed entries against the reference before reporting.

| Feature type | Metric in the workbook | Interpretation |
| --- | --- | --- |
| Numerical | NRMS = Frobenius norm of the difference divided by the Frobenius norm of the complete numerical matrix | Lower is better. Uses original units and includes observed entries in the denominator, as specified in the course instructions. Large-magnitude features can dominate. |
| Categorical | AE = correctly imputed hidden categories divided by the number of hidden categories | Higher is better. AE uses exact category matches and ranges from 0 to 1. |
| Mixed | Both metrics, each on its own columns | A mask with no hidden categorical entries has undefined AE, recorded as N/A. |

Blank metric cells indicate an inapplicable feature type. The workbook averages give each scored variant equal weight. They combine different masking rates, feature scales and datasets.

The numerical group follows the course inventory, including integer-valued features the inventory treats as numerical. The categorical group includes TTTTEG (TTTEG in the template), HOV, MUSH, Splice and C4. The mixed group includes Aheart, Credit, Abalone, Adult and KDD. Credit’s actual data have six numerical and nine text columns, reversing the inventory counts. HOV contains 232 rows in the workbooks, compared with 234 in the inventory. Splice’s complete workbook pads DNA symbols with spaces, which the scorer trims for comparison.

### Completed sweep

The workbook contains 1,144 independent fits. The mean NRMS was **0.135923** over 924 numerical or mixed variants. Mean AE was **0.607419** over 406 categorical or mixed variants with at least one hidden category. The remaining 34 mixed masks contained only hidden numerical cells.

| Nominal missingness | Mean NRMS | Mean AE | Variants contributing to AE |
| --- | ---: | ---: | ---: |
| 1% | 0.04953 | 0.61168 | 100 |
| 5% | 0.11298 | 0.62403 | 99 |
| 10% | 0.15083 | 0.59893 | 104 |
| 20% | 0.23036 | 0.59589 | 103 |

Each rate contributes 231 NRMS scores. Average NRMS increased at every rate, consistent with more cells contributing reconstruction error against a fixed reference norm. AE varied less across rates. Its contributing masks also vary because some mixed variants have no hidden categories.

### Training budget and scalability

The recorded sweep uses Adam (0.002), two 32-unit hidden layers, four latent dimensions, batch size 256, four training importance samples, eight imputation samples and seed 1234. Each model trains on all rows for whole shuffled epochs until it has completed at least one epoch and 64 updates. This gives smaller datasets more passes and makes the full sweep feasible on CPU. These settings are a reproducible first benchmark. Convergence and hyperparameter selection were not evaluated.

For fixed architecture, training time grows with epochs × rows × training importance samples. Imputation time grows with rows × imputation samples. Decoder work also increases with the total number of category logits. Minibatches bound neural-network activation memory. The loader and evaluation retain complete and incomplete tables in memory, so the overall memory requirement still grows with dataset size. KDD is the largest case at 494,020 rows and 41 features.

## Original Iris tutorial

### Tutorial files

| File | Purpose |
| --- | --- |
| [MIWAE_Pytorch_exercises_demo_ProbAI.ipynb](MIWAE_Pytorch_exercises_demo_ProbAI.ipynb) | Data preparation, model definitions, training, imputation, baseline comparisons, and an extension exercise. |
| [requirements.txt](requirements.txt) | Python dependencies. |
| [LICENSE](LICENSE) | MIT license. |

## Experiment

The default experiment uses the Iris dataset bundled with scikit-learn: **150 observations and 4 numerical features**. Class labels are not used.

1. Standardize each feature using the complete dataset.
2. Remove exactly 50% of entries uniformly at random, simulating missing completely at random (MCAR) data.
3. Fill missing entries with zero for the encoder input and retain a mask identifying observed values.
4. Train MIWAE using the likelihood of observed entries.
5. Estimate missing values using self-normalized importance sampling.
6. Compare mean squared error (MSE) on the artificially removed entries with the baseline methods.

The model uses a standard Gaussian latent prior, a Gaussian approximate posterior, and a Student's t observation distribution. Both the encoder and decoder have two hidden layers with ReLU activations.

### Default settings

| Setting | Value |
| --- | --- |
| Missing fraction (`perc_miss`) | `0.5` |
| Hidden units per layer (`h`) | `128` |
| Latent dimensions (`d`) | `1` |
| Importance samples during training (`K`) | `20` |
| Optimizer | Adam, learning rate `1e-3` |
| Batch size parameter (`bs`) | `64` |
| Training iterations | 2,001 epochs (`range(1, 2002)`) |
| Evaluation interval | Every 100 epochs, starting at epoch 1 |
| Importance samples during evaluation (`L`) | `10` |

The training loop splits the data into `max(1, n // bs)` batches; actual batch sizes can therefore differ from `bs`.

## Setup

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/mohiburnabil/CS6725_MIWAE.git
cd CS6725_MIWAE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install notebook ipykernel
python -m ipykernel install --user --name miwae --display-name "Python (MIWAE)"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The dependencies include PyTorch, torchvision, NumPy, SciPy, pandas, Matplotlib, and scikit-learn. Versions are not pinned. Jupyter and the kernel package are installed separately by the commands above.

### Device requirements

**The original ProbAI tutorial requires CUDA to run unchanged.** It contains explicit `.cuda()` calls in the prior, model setup, and imputation function. The later automatic device selection does not override these calls. Use a CUDA-capable NVIDIA GPU with a compatible PyTorch installation, or upload the notebook to Google Colab and select a GPU runtime.

For CPU execution, define the device before constructing the prior:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Then replace **every** `.cuda()` call with `.to(device)` and rerun the notebook from the beginning. This also allows CPU execution on Macs without CUDA.

## Run the notebook

```bash
jupyter notebook MIWAE_Pytorch_exercises_demo_ProbAI.ipynb
```

Select the **Python (MIWAE)** kernel and run the cells in order. The notebook also works in another Jupyter-compatible editor when using the same environment.

The first code cell includes `!pip3 install --user --upgrade scikit-learn`. If you installed the dependencies in the virtual environment above, skip or remove that line to avoid an additional user-level installation.

During training, the notebook prints the MIWAE likelihood bound and imputation MSE. The final plot compares the MIWAE MSE trajectory with:

- An Extra Trees iterative imputer, used as a missForest-like baseline and labeled “missForest” in the plot.
- An iterative imputer using Bayesian ridge regression.
- Feature-wise mean imputation.

Lower MSE indicates better recovery of the removed values. Results and plots appear inside the notebook; it does not automatically save model checkpoints or export imputed datasets.

## Customize the experiment

- Change `perc_miss` to test another missing-data fraction.
- Adjust `h`, `d`, `K`, `bs`, or `n_epochs` to explore model capacity and training behavior.
- Change `L` in the evaluation call to adjust the number of imputation samples.
- Replace the Iris loading cell with another numerical data matrix. Commented examples include breast cancer, wine quality, and banknote authentication datasets; URL-based examples require internet access.

The commented dataset examples may need updates before use. The default Iris path uses `load_iris(return_X_y=True)` and needs no external data download.

## Interpretation and reproducibility

This is a demonstration on one dataset and one missingness pattern. Standardization uses the complete data before masking, and training and evaluation use the same rows. The reported MSE measures reconstruction of hidden entries in standardized units; it is not a held-out generalization benchmark.

NumPy is seeded with `1234` before generating the missingness pattern. PyTorch and the randomized tree baseline are not explicitly seeded, so results can vary across runs. The notebook ends with an exercise to improve the single-imputation estimator.

## Reference

Pierre-Alexandre Mattei and Jes Frellsen. **MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets.** Proceedings of the 36th International Conference on Machine Learning, PMLR 97:4413–4423, 2019. [Paper on arXiv](https://arxiv.org/abs/1812.02633).

## License

This repository is distributed under the [MIT License](LICENSE).
