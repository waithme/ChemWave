# ChemWave: code and data

ChemWave is a molecular graph-learning model for assay-specific activity prediction, with an atom-centred role–distance representation. This repository is intentionally limited to source code, input datasets, dependency specifications, and this project README. It does not distribute manuscript files, figures, experiment outputs, trained weights, logs, or review/archive snapshots.

## Repository contents

| Path | Contents |
|---|---|
| `final_model/` | Final ChemWave model, training/evaluation entry points, and unit tests. |
| `ablation_experiment/` | Architecture ablation implementation and input assay data. |
| `ablation_results/` | Local-only generated results; excluded from version control. |
| `experiment3/` | Training-strategy comparisons and tests. |
| `experiment4/` | Prediction-intervention / interpretability analysis code. |
| `matched_multitarget_baselines/` | Matched GINE and AttentiveFP baseline implementations. |
| `e2_assay_film_gine_control/` | Matched assay-FiLM GINE control implementation and tests. |
| `chemical_semantic_control/` | Chemical-role versus element-shell control code. |
| `sensitivity_radius/`, `sensitivity_shell/`, `sensitivity_normalization/` | Representation sensitivity analyses. |
| `target_balanced_experiment/` | Target-balanced versus row-weighted multi-task training comparison. |
| `efficiency_experiment/` | Runtime and parameter-count measurement code. |
| `v1/`, `v1_relative_position/` | Earlier model implementations retained for reproducibility. |
| `*/data/` | Input assay data required by the corresponding scripts. The legacy copies are retained because the scripts use directory-relative data paths. |
| `*/requirements.txt` | Python package requirements for the corresponding experiment. |

Generated result tables and JSON summaries are deliberately not committed. In particular, the E2 control's reported outputs are not included here; `e2_assay_film_gine_control/` contains its code and tests only. Model checkpoints and trained weights are also excluded.

## Environment and dependencies

Experiments have separate `requirements.txt` files because their environments evolved independently. Use the file in the experiment directory you intend to run; avoid combining all experiment environments unless you have resolved version conflicts. The E2 assay-FiLM GINE control was tested with RDKit 2023.09.6 and PyTorch 2.5.1. GPU runs require a CUDA-compatible PyTorch installation matching the host driver.

Example setup for the final model (use an isolated environment):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r final_model/requirements.txt
```

## Running experiments

Run each entry point from its own directory so its relative imports and paths resolve as expected. Training can take substantial time and will create checkpoints and result files locally; these generated artifacts are ignored by Git.

```bash
# Final ChemWave (training/evaluation)
cd final_model
python run_final_model.py --help

# Architecture ablations
cd ../ablation_experiment
python run_ablation.py --help

# Training-strategy comparison
cd ../experiment3
python run_experiment3.py --help

# Matched GINE / AttentiveFP controls
cd ../matched_multitarget_baselines
python run_all.py --help

# Assay-FiLM GINE control (supply an input-data path available on this machine)
cd ../e2_assay_film_gine_control
python run_all.py --help
```

Use the command-line `--help` output for the exact arguments supported by each checked-in script. Do not evaluate or select models using the official test set; preserve the validation-based checkpoint-selection protocol implemented by each experiment.

## Tests

The repository includes focused unit tests for model components and controls. Install the experiment's requirements, then run tests from that experiment directory, for example:

```bash
cd final_model
python -m pytest -q tests

cd ../e2_assay_film_gine_control
python -m pytest -q tests
```

## Data and reproducibility notes

The checked-in `data/` directories contain input assay records used by the experiments; derived benchmark summaries and per-seed predictions are excluded. The same assay files are present in a few legacy locations because older scripts expect those paths. Treat them as copies of the same inputs, not independent datasets.

For reproducibility, record the experiment directory, command-line arguments, Python/package versions, device, and random seeds when running a job. Checkpoint selection should use only the designated validation data, with test evaluation performed after selection. The project does not bundle trained checkpoints or precomputed results.

## Scope of this GitHub copy

Included: computational source code, input data directories, experiment-specific dependency files, and this README. Excluded: manuscript and Supporting Information source/PDFs, figures, generated results, trained model weights, logs/caches, dated review packages, and retired/archive files. The excluded material remains in the local project folder and is not removed by this repository setup.
