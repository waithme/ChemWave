# Saved intervention audit data

This release supports numeric verification of main Figure 4, the selected
main Figure 5 molecule, and the Supporting Information molecular gallery.
It preserves all 1,476 sampled molecules from 30 assays and seeds 0--4, not
only the 120 displayed cases. No model inference or GPU is needed to use it.

## Files and definitions

| File | Contents |
|---|---|
| `molecule_scores.csv.gz` | 7,380 molecule--seed records: assay, zero-based source CSV row, SMILES, observed activity, cliff flag, seed, and prediction. |
| `atom_scores.csv.gz` | 227,830 atom--seed records: assay, source row, seed, atom index, and signed relative-field intervention. |
| `bond_scores.csv.gz` | 250,695 bond--seed records: assay, source row, seed, bond index, endpoint atom indices, and signed bond-difference intervention. |
| `seed_stability.csv.gz` | 1,476 per-molecule stability summaries, including absolute-magnitude ranks, archived Top-3 overlaps, and separately named signed diagnostics. |
| `case_selection.csv` | The original 120 case selections, input hashes, output routing names, and main/SI location. The main Figure 5 case is not duplicated in the 119-image SI gallery. |
| `figure5_summary.json` | Five per-seed predictions and highlighted component effects, their means, and the printed three-decimal annotations. |
| `manifest.json` | Scope, schemas, source and compact-file hashes, sampling settings, and provenance limitations. |
| `reaggregation_audit.json` | A no-inference re-aggregation of the compact tables using the released analysis code. |
| `checksums.sha256` | SHA-256 checksums for this release's files other than the checksum list itself. |

Every retained value is copied as its original CSV string, without numeric
rounding. SMILES, activities, and flags are stored in the molecule table rather
than repeated in each atom/bond row; unrelated internal diagnostics are omitted.
These changes reduce the four tables from about 82.4 MiB to about 5.3 MiB with
lossless gzip compression. A row is identified by `(target, row_id, seed)`;
`row_id` is the zero-based row in the assay input CSV, not its position within
the test subset. Atom/bond indices refer to the saved SMILES parsing order.

An atom's `relative_prediction_delta` is the full prediction minus the
prediction after zeroing only that atom's relative-position field. A bond's
`gradient_prediction_delta` is the full prediction minus the prediction after
masking that chemical bond only in the bond-difference branch; ordinary graph
messages remain. The legacy word `gradient` in the column name denotes this
same intervention, not a new derivative-based attribution method.

Figure 4 ranks absolute effects within each seed, uses average ranks for ties
in Spearman correlation, and averages the ten seed-pair correlations per
molecule. Its across-molecule means are about 0.256824 for atoms and 0.209607
for bonds (printed as 0.257 and 0.210). Signed Spearman columns are separate
directional diagnostics. The existing utility retains, rather than recomputes,
the archived Top-3 overlap columns. Re-aggregated floating-point CSV bytes can
differ across library versions even when values agree to numerical precision.

Case colors average signed effects across the five seeds before applying
separate molecule-local atom/bond scales. Red is positive and blue negative.
Signed averaging can cancel opposite directions. Neither channel is an additive
prediction decomposition or a validated causal explanation.

## Verify and extract

Run from the repository root. Verify the release with either Linux's
`sha256sum` or macOS's `shasum`:

```bash
cd publication_data/interventions
sha256sum -c checksums.sha256
# macOS alternative: shasum -a 256 -c checksums.sha256
cd ../..
```

Extract into a new temporary directory, keeping the distributed files intact.
This uses only the Python standard library:

```bash
export CHEMWAVE_AUDIT_DIR="$(mktemp -d)"
python - "$CHEMWAVE_AUDIT_DIR" <<'PY'
import csv
import gzip
import shutil
import sys
from pathlib import Path

source = Path('publication_data/interventions')
destination = Path(sys.argv[1])
for archive in sorted(source.glob('*.csv.gz')):
    with gzip.open(archive, 'rb') as src:
        with (destination / archive.stem).open('xb') as dst:
            shutil.copyfileobj(src, dst)
with (source / 'case_selection.csv').open(newline='') as stream:
    cases = list(csv.DictReader(stream))
assert len(cases) == 120
selection = ''.join(
    '\\includegraphics{figures/molecule_cases/' + case['relative_image'] + '}\n'
    for case in cases
)
(destination / 'case_selection.tex').write_text(selection)
print(destination)
PY
```

The generated routing file contains only the 120 selected image paths. It is
not a copy of the manuscript or SI source.

## Recompute Figure 4 values

Install the analysis dependencies listed in `experiment4/requirements.txt`,
then run the existing utility from the repository root:

```bash
python experiment4/recompute_magnitude_seed_stability.py \
  --tables-dir "$CHEMWAVE_AUDIT_DIR"
```

It checks all five seeds and component indices, validates signed diagnostics,
and writes `seed_stability_magnitude.csv` and an output audit in the temporary
directory. Original inputs are not overwritten. Compare the resulting means
with `reaggregation_audit.json`; this checks the published numeric statistics,
not the exact typesetting of Figure 4.

## Verify Figure 5 and redraw case colors

The main example is `CHEMBL287_Ki`, source row 888. In
`figure5_summary.json`, badges A, B, and C map to atom indices 10, 14, and 15;
the highlighted bond has index 14 and connects atoms 14 and 15. Its per-seed
values can also be selected directly from the three compact score tables.

To redraw all 120 selected cases with the existing renderer:

```bash
python experiment4/regenerate_case_gallery_from_saved_scores.py \
  --gallery-tex "$CHEMWAVE_AUDIT_DIR/case_selection.tex" \
  --tables-dir "$CHEMWAVE_AUDIT_DIR" \
  --output-dir "$CHEMWAVE_AUDIT_DIR/gallery"
```

For a four-case smoke test, append `--target CHEMBL287_Ki`. The selected main
case is generated alongside that assay's SI cases; `case_selection.csv`
identifies its main-text location. This recreates numeric labels and color
maps, not the final two-panel vector composition and A--C badges of Figure 5.
The latest manuscript gallery was depicted with RDKit 2026.03.1, separately
from the RDKit 2023.09.6 training environment. Other depiction versions may
place atoms, labels, or highlights differently without changing saved scores.
Images generated by these commands stay in the temporary output directory and
are not part of this repository's data release.

## Provenance and limits

`manifest.json` records the hashes of the original local CSVs and the compact
exports, the analysis code revision, and the hashes of the analysis scripts.
Case input-file/row hashes bind the selected examples to the public assay
inputs. Published hashes identify data, not an independent certification that
all historical training metadata can be reconstructed.

The original saved intervention run manifest does not embed per-output
checkpoint hashes. We do not retrospectively invent them. Model weights are
not distributed here, so this release enables re-aggregation and case-color
audits from saved outputs, not independent generation of those outputs from
trained models. It contains no external-baseline results, general training
logs, manuscript files, images, or full publication-figure pipeline.
