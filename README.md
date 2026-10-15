# ChemWave: code and data

ChemWave is a molecular graph-learning model for assay-specific activity prediction, with an atom-centred role–distance representation. This repository contains source code, input datasets, dependency specifications, and a compact saved-intervention audit dataset. It does not distribute manuscript files, figures, general experiment outputs, trained weights, logs, or review/archive snapshots.

## Repository contents

| Path | Contents |
|---|---|
| `final_model/` | Final ChemWave model, training/evaluation entry points, and unit tests. |
| `ablation_experiment/` | Architecture ablation implementation and input assay data. |
| `experiment3/` | Training-strategy comparisons and tests. |
| `experiment4/` | Prediction-intervention / interpretability analysis code. |
| `publication_data/interventions/` | Losslessly compressed saved predictions and atom/bond interventions for all 1,476 sampled molecules and five seeds; Figure 4 statistics, Figure 5 numeric checks, and the 120-case gallery selection. See its README for extraction and verification commands. |
| `matched_multitarget_baselines/` | Matched GINE and AttentiveFP baseline implementations. |
| `e2_assay_film_gine_control/` | Matched assay-FiLM GINE control implementation and tests. |
| `chemical_semantic_control/` | Chemical-role versus element-shell control code. |
| `sensitivity_radius/`, `sensitivity_shell/`, `sensitivity_normalization/` | Representation sensitivity analyses. |
| `target_balanced_experiment/` | Target-balanced versus row-weighted multi-task training comparison. |
| `efficiency_experiment/` | Runtime and parameter-count measurement code. |
| `v1/`, `v1_relative_position/` | Earlier implementations retained as development history; their JSON lock files are not distributed. |
| `*/data/` | Input assay data required by the corresponding scripts. The legacy copies are retained because the scripts use directory-relative data paths. |
| `*/requirements.txt` | Python package requirements for the corresponding experiment. |

The explicit exception to the generated-output exclusion is `publication_data/interventions/`: it contains selected columns from the saved intervention run, not a selected subset of molecules. All 1,476 sampled molecules, five seeds, and retained numeric strings are preserved. Other generated benchmark tables and summaries remain excluded. In particular, the E2 control's reported outputs are not included here; `e2_assay_film_gine_control/` contains its code and tests only. Model checkpoints and trained weights are also excluded.

Experiment directories retain local module snapshots so that each launcher can run from its own directory. `final_model/` is the canonical Full implementation. File headers identify snapshots with unchanged feature rules and deliberate architecture, semantic-control, sampling, or representation-sensitivity forks. Changes do not automatically propagate to these copies. The feature and model files in `experiment4/` must remain byte-identical to their canonical counterparts for checkpoint source-hash verification, including comments; their canonical headers are therefore retained verbatim. This layout does not imply additional final models.

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

New runs embed an effective architecture/training protocol fingerprint in each checkpoint. Adapted checkpoints also bind the exact shared-pretraining checkpoint SHA-256. Result rows record both checkpoint hashes, both protocol fingerprints, the test-input hash, and the evaluation-script hash. A checkpoint/configuration mismatch or a same-target result with different provenance stops the run; historical CSVs/checkpoints are not silently upgraded or overwritten. Use a fresh output directory for a new protocol. The historical fingerprint above is a property of the archived results, not a guarantee that a different local environment or input snapshot will recompute it.

In the ablation code, final Full is `a5_full` (the constructor default), and
the affine-transport comparator is `a3_affine_transport`. Historical
`a5_plain_bond_gradient` and `a3_full` remain compatibility aliases for these
respective variants. The standalone final-model provenance identifier stays
`a5_plain_bond_gradient`; existing checkpoint and result identities are not
rewritten. The seven canonical variants are:

| Code identifier | Model / change |
|---|---|
| `a0_v1` | Base aggregation/contrast model; no relative field or bond-difference response |
| `a1_relative` | Assay-gated relative field only |
| `a2_bond` | Assay-conditioned bond-difference response only |
| `a3_affine_transport` | Combined fields with affine neighbor transport; comparator, not final Full |
| `a4_no_target_relative` | Full with a shared, ungated relative field |
| `a5_full` | Final Full ChemWave; direct center-minus-neighbor response, no affine transport |
| `a6_no_target_bond` | Full with a shared bond response |

From the repository root, the other main experiment launchers can be run as follows after setting `CHEMWAVE_DATA_DIR`:

```bash
# One architecture ablation; repeat with each variant listed by --help.
cd ablation_experiment
python run_ablation.py --variant a5_full \
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

The remaining published controls can be launched from the repository root with the same input directory and five-seed protocol. Each sensitivity variant is a separate run; `runs/` and `final_runs/` contain generated outputs and are excluded from Git:

```bash
cd chemical_semantic_control
for variant in element_shell random_role; do
  python run_semantic_control.py --variant "$variant" \
    --data-dir "$CHEMWAVE_DATA_DIR" --output-root runs --device cuda
done

cd ../sensitivity_radius
for variant in radius_1 radius_3 radius_5; do
  python run_sensitivity.py --variant "$variant" \
    --data-dir "$CHEMWAVE_DATA_DIR" --output-root runs --device cuda
done

cd ../sensitivity_shell
for variant in exact_shell cumulative_shell; do
  python run_sensitivity.py --variant "$variant" \
    --data-dir "$CHEMWAVE_DATA_DIR" --output-root runs --device cuda
done

cd ../sensitivity_normalization
for variant in norm_none norm_mean norm_sqrt; do
  python run_sensitivity.py --variant "$variant" \
    --data-dir "$CHEMWAVE_DATA_DIR" --output-root runs --device cuda
done

cd ../target_balanced_experiment
python run_final_model.py --data-dir "$CHEMWAVE_DATA_DIR" \
  --output-dir final_runs --device cuda

cd ../efficiency_experiment
python run_efficiency.py --data-dir "$CHEMWAVE_DATA_DIR" \
  --output-dir runs/efficiency --device cuda --seed 0
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

Use each launcher's `--help` for additional options. Official test metrics are for reporting after validation-based model and checkpoint selection; they are not selection criteria.

Interpretability runs verify the development-data fingerprint and every requested checkpoint before generating explanations, then record checkpoint and input hashes in their manifest. Historical checkpoints without embedded protocol metadata require a separate read-only audit manifest via `--legacy-audit-manifest`; that manifest documents unresolved provenance limits rather than treating old weights as newly certified. Use `--verify-only` to check sources without producing explanations.

### Cross-seed intervention ranking

The Figure 4 ranking statistics use the absolute intervention magnitudes
`abs(full_prediction - masked_prediction)`, for both Spearman correlation and
Top-3 overlap. Positive and negative prediction changes with the same magnitude
therefore receive the same rank. Per molecule, the analysis averages the ten
pairs of five model seeds. Signed Spearman correlations are retained only as
separate directional diagnostics in `atom_signed_spearman_mean` and
`bond_signed_spearman_mean`; they are not the main ranking-agreement statistic.

Earlier repository revisions ranked signed effects in the main Spearman
columns. New intervention runs use the corrected absolute-magnitude definition.
If you already generated the publication-scale saved intervention tables,
recompute the ranking summaries without loading checkpoints or rerunning the
model:

```bash
cd experiment4
python recompute_magnitude_seed_stability.py
cd ..
```

This study-specific utility expects `atom_scores.csv`, `bond_scores.csv`, and
`seed_stability.csv` under `experiment4/interpretability_outputs/tables/`, with
the 1,476 sampled molecules, 30 assays, and seeds 0–4. A different input location
can be supplied with `--tables-dir /path/to/tables`. It validates component
indices and signed diagnostics, then writes `seed_stability_magnitude.csv`
and `seed_stability_magnitude_output_audit.json`, recording input/output hashes.
The input CSVs and historical audit files remain intact. Identical output files
can be reused; different existing outputs are not overwritten. Signed
diagnostics and the saved Top-3 columns are retained. No model inference is
performed. Compact versions of the required saved tables are now distributed in
`publication_data/interventions/`. Follow that directory's README to extract them
to a temporary directory, then pass that directory via `--tables-dir` to verify
the published absolute-magnitude means without model inference. The complete
original run directory and trained checkpoints remain excluded.

### Figure diagnostics and gallery rendering

`experiment4/interpretability_visuals.py` renders quick-look diagnostics, not the
publication statistical-figure layouts. Its stability diagnostic separates
Spearman correlations (including negative values) from Top-3 overlaps. The
manuscript's figures and figure-specific assembly scripts are outside this
code/data repository.

`experiment4/regenerate_case_gallery_from_saved_scores.py` preserves the supplied
case selection and redraws its molecules from saved five-seed component effects,
including the mean-prediction label and separate local atom/bond maxima. The
compact saved tables and a machine-readable 120-case selection are now in
`publication_data/interventions/`; its README shows how to generate the temporary
selection routing file accepted by this existing renderer. This does not require
the manuscript TeX. For other locally available inputs, supply their paths:

```bash
cd experiment4
python regenerate_case_gallery_from_saved_scores.py \
  --gallery-tex /path/to/molecule_gallery.tex \
  --tables-dir /path/to/saved/tables \
  --output-dir /path/to/new/gallery
cd ..
```

The default depiction size is 1200 by 800 pixels before adding the legend. This
changes only rendering; it does not run the model, alter saved scores, select
new cases, or make a chemical-causality claim.

## Tests

The repository includes focused unit tests for model components and controls.
Install the experiment's requirements and `requirements-dev.txt`, then run tests
from that experiment directory, for example:

```bash
python -m pip install -r requirements-dev.txt
cd final_model
python -m pytest -q tests
python -m unittest discover -s tests -t .

cd ../e2_assay_film_gine_control
python -m pytest -q tests
```

Run each experiment suite in a separate process from its own directory; the local
module imports intentionally resolve to that experiment's snapshot. The test
packages support `unittest` discovery with `-t .`. `pytest.ini` limits filename
collection to `test_*.py`, and production `test_metrics()` helpers are explicitly
excluded from pytest collection without changing their evaluation behavior.

## Data and reproducibility notes

The checked-in experiment `data/` directories contain input assay records. The separate `publication_data/interventions/` release contains saved predictions and component effects for the sampled intervention analysis only; derived external-benchmark summaries remain excluded. The same assay files are present in a few legacy locations because older scripts expect those paths. Treat them as copies of the same inputs, not independent datasets.

For reproducibility, record the experiment directory, command-line arguments, Python/package versions, device, and random seeds when running a job. Checkpoint selection uses only the designated development validation data, with official-test evaluation performed afterward. The project does not bundle trained checkpoints. The compact intervention release supports re-aggregation and numeric case audits, not independent regeneration of model outputs from released weights.

The `v1/` and `v1_relative_position/` directories are archival code, not runnable publication entry points in this public checkout. Their historical `CHEMWAVE_MODEL_LOCK.json` files, and `v1/CHEMWAVE_EVALUATION_REGISTRY.json`, are not distributed. To reproduce the Base (V1) architecture under the final five-seed ablation protocol, use `ablation_experiment/run_ablation.py --variant a0_v1`; do not treat the archived V1 launchers as a second final model.

## Scope of this GitHub copy

Included: computational source code, input data directories, experiment-specific dependency files, this README, and the explicitly curated saved-intervention audit dataset. Excluded: manuscript and Supporting Information source/PDFs, figures, other generated results, trained model weights, logs/caches, dated review packages, and retired/archive files. The excluded material remains in the local project folder and is not removed by this repository setup.
