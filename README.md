# HEARSAY - HackGT 13

## HEARSAY evaluation

The canonical evaluator is `evaluation.min_dcf.evaluate_min_dcf`. It uses the
NSA cost model for ASVspoof5 Track 1:

- `Pspoof = 0.30`
- `Cmiss = 1.0`
- `Cfa = 4.0`

HEARSAY detector scores are synthetic-oriented: larger means more likely
synthetic/spoof. ASVspoof's countermeasure DET calculation is bona-fide-oriented,
so the evaluator explicitly negates scores internally and converts the selected
threshold back to the supplied synthetic-score orientation.

Lower minDCF is better. The metric sweeps all score thresholds, so ranking and
class separation matter more than whether a score is calibrated around `0.5`.
In the countermeasure cost model, `Cfa` applies to spoof samples falsely accepted
as bona fide.

Evaluate header-based tab-separated files by ID:

```bash
python -m evaluation.evaluate_scores \
  --scores predictions.tsv \
  --labels answer_key.tsv \
  --id-column id \
  --score-column score \
  --label-column label
```

Use `--delimiter comma` for CSV input. Labels must be exactly `bonafide` or
`spoof`; predictions and labels are joined by file ID rather than row position.
