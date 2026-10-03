# ChemWave: code and data

ChemWave is a molecular graph-learning model for assay-specific activity prediction, with an atom-centred role–distance representation. This repository is intentionally limited to source code, input datasets, dependency specifications, and this project README. It does not distribute manuscript files, figures, experiment outputs, trained weights, logs, or review/archive snapshots.

## Repository contents

| Path | Contents |
|---|---|
| `final_model/` | Final ChemWave model, training/evaluation entry points, and unit tests. |
| `ablation_experiment/` | Architecture ablation implementation and input assay data. |
| `experiment3/` | Training-strategy comparisons and tests. |
| `experiment4/` | Prediction-intervention / interpretability analysis code. |
| `matched_multitarget_baselines/` | Matched GINE and AttentiveFP baseline implementations. |
| `e2_assay_film_gine_control/` | Matched assay-FiLM GINE control implementation and tests. |
| `chemical_semantic_control/` | Chemical-role versus element-shell control code. |
| `sensitivity_radius/`, `sensitivity_shell/`, `sensitivity_normalization/` | Representation sensitivity analyses. |
| `target_balanced_experiment/` | Target-balanced versus row-weighted multi-task training comparison. |
| `efficiency_experiment/` | Runtime and parameter-count measurement code. |
| `v1/`, `v1_relative_position/` | Earlier implementations retained as development history; their JSON lock files are not distributed. |
| `*/data/` | Input assay data required by the corresponding scripts. The legacy copies are retained because the scripts use directory-relative data paths. |
| `*/requirements.txt` | Python package requirements for the corresponding experiment. |

Generated result tables and JSON summaries are deliberately not committed. In particular, the E2 control's reported outputs are not included here; `e2_assay_film_gine_control/` contains its code and tests only. Model checkpoints and trained weights are also excluded.

## Environment and dependencies

Use Python 3.10.20 and the `requirements.txt` in the experiment directory you intend to run. The core versions reported in the Supporting Information are NumPy 1.26.4, pandas 2.3.3, RDKit 2023.09.6, PyTorch 2.5.1+cu124, and PyTorch Geometric 2.7.0; these core versions are pinned in the checked-in requirement files. The intervention plotting requirement is pinned to Matplotlib 3.10.9 for repeatable rendering; its historical publication-run version was not recorded. The requirement files are direct-dependency pins, not a full transitive environment lock. GPU users need a PyTorch wheel compatible with their driver; the following commands reproduce the reported CUDA 12.4 wheel selection on Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r final_model/requirements.txt
python -c 'import numpy, pandas, rdkit, torch, torch_geometric; print(numpy.__version__, pandas.__version__, rdkit.__version__, torch.__version__, torch_geometric.__version__)'
```

## Running experiments

The canonical inputs are the 30 MoleculeACE CSVs in `ablation_experiment/data/experiment2`. Run each entry point from its own directory so relative imports resolve. The commands below use seeds 0–4, hidden width 300, at most 100 epochs per training stage, and the scripts' validation-based early stopping. Set the data path once from the repository root:

```bash
export CHEMWAVE_DATA_DIR="$PWD/ablation_experiment/data/experiment2"
```

Final Full ChemWave, including joint training, target adaptation, and one subsequent official-test evaluation per target and seed:

```bash
cd final_model
python run_final_model.py \
  --data-dir "$CHEMWAVE_DATA_DIR" \
  --output-dir final_runs \
  --device cuda \
  --hidden-dim 300 \
  --pretrain-epochs 100 \
  --finetune-epochs 100 \
  --expected-target-count 30
cd ..
```

This writes `final_model/final_runs/results/pretrain.json` and `final_model/final_runs/results/all_results.csv`. The expected final table has 150 unique target–seed rows (30 targets × five seeds). Its development-data fingerprint is `b9deaeb5c67b1b7aee83f0e5c5a48bd85b9d74e1865e13d26ca8ef6224ee7184`.

From the repository root, the other main experiment launchers can be run as follows after setting `CHEMWAVE_DATA_DIR`:

```bash
# One architecture ablation; repeat with each variant listed by --help.
cd ablation_experiment
python run_ablation.py --variant a5_plain_bond_gradient \
  --data-dir "$CHEMWAVE_DATA_DIR" --output-root runs --device cuda

# From-scratch, joint-only, and joint-plus-adaptation protocols.
cd ../experiment3
python run_experiment3.py --data-dir "$CHEMWAVE_DATA_DIR" \
  --output-dir experiment3_runs --device cuda

# Matched multi-target GINE and AttentiveFP controls.
cd ../matched_multitarget_baselines
python run_all.py --model-name gine --data-dir "$CHEMWAVE_DATA_DIR" \
  --output-dir matched_runs --device cuda
python run_all.py --model-name attentivefp --data-dir "$CHEMWAVE_DATA_DIR" \
  --output-dir matched_runs --device cuda

# Matched assay-FiLM GINE control.
cd ../e2_assay_film_gine_control
python run_all.py --data-dir "$CHEMWAVE_DATA_DIR" \
  --output-dir matched_runs --device cuda
cd ..
```

For the publication-scale intervention analysis, run the final model first and then use its adapted checkpoints. The script's lightweight defaults are 10 cliff molecules, 10 non-cliff molecules, and five random repeats; the reported protocol uses 25/25/10. The script writes these resolved settings to `interpretability_outputs/manifest.json`.

```bash
cd experiment4
python -m pip install -r requirements.txt
python run_interpretability.py \
  --data-dir "$CHEMWAVE_DATA_DIR" \
  --checkpoint-dir ../final_model/final_runs/checkpoints/adapted \
  --output-dir interpretability_outputs \
  --seeds 0 1 2 3 4 \
  --max-cliff 25 --max-noncliff 25 --random-repeats 10 \
  --render-cases-per-target 4 --device cuda
cd ..
```

The architecture ablation, chemical-role control, radius/shell/normalization sensitivity, target-balanced, and efficiency directories each have their own launchers. Use `--help` for their variant choices and pass the same `CHEMWAVE_DATA_DIR`, seeds, and published training budget. Official test metrics are for reporting after validation-based model and checkpoint selection; they are not selection criteria.

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

For reproducibility, record the experiment directory, command-line arguments, Python/package versions, device, and random seeds when running a job. Checkpoint selection uses only the designated development validation data, with official-test evaluation performed afterward. The project does not bundle trained checkpoints or precomputed results.

The `v1/` and `v1_relative_position/` directories are archival code, not runnable publication entry points in this public checkout. Their historical `CHEMWAVE_MODEL_LOCK.json` files, and `v1/CHEMWAVE_EVALUATION_REGISTRY.json`, are not distributed. To reproduce the Base (V1) architecture under the final five-seed ablation protocol, use `ablation_experiment/run_ablation.py --variant a0_v1`; do not treat the archived V1 launchers as a second final model.

## Scope of this GitHub copy

Included: computational source code, input data directories, experiment-specific dependency files, and this README. Excluded: manuscript and Supporting Information source/PDFs, figures, generated results, trained model weights, logs/caches, dated review packages, and retired/archive files. The excluded material remains in the local project folder and is not removed by this repository setup.
