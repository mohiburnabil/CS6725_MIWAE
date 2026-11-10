# CS6725: Missing Data Imputation with MIWAE

A PyTorch notebook for studying the **Missing-data Importance-Weighted Autoencoder (MIWAE)** in a CS6725 data mining project. The notebook learns a deep generative model from incomplete continuous data, imputes missing values, and compares imputation accuracy with conventional methods.

The implementation follows the MIWAE tutorial described in the notebook and the paper by Pierre-Alexandre Mattei and Jes Frellsen, *MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets* (ICML 2019).

## Repository contents

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

**The notebook currently requires CUDA to run unchanged.** It contains explicit `.cuda()` calls in the prior, model setup, and imputation function. The later automatic device selection does not override these calls. Use a CUDA-capable NVIDIA GPU with a compatible PyTorch installation, or upload the notebook to Google Colab and select a GPU runtime.

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
