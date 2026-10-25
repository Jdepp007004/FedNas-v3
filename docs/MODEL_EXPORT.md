# Downloading and using a trained tabular model

After a self-serve tabular workspace completes at least one round, the host
dashboard enables **Download latest model bundle**. The ZIP contains:

- the latest completed global-model weights as a compressed NumPy archive;
- the model and preprocessing configuration, including feature columns,
  numeric scaling values, target mapping, and class labels;
- a manifest with the project ID and trained round;
- a standalone `predict_tabular.py` script and usage notes.

The bundle does not include participant training CSVs. Extract it, place a new
CSV containing the model's feature columns in the extracted directory, and
run:

```bash
python predict_tabular.py --input new_rows.csv --output predictions.csv
```

Install Python 3.10 or later with `numpy`, `pandas`, and `torch`. The output
keeps the input columns and adds a `prediction` column. Classification also
adds one probability column per class; regression outputs are converted back
to the target's original units.

The export is available for self-serve tabular projects. It represents the
full coordinator model, so review the separate full-model validation metric
before using it. This capstone tool is not a clinical decision system.
