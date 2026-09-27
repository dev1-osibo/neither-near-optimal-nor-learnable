# Figure reproduction

The eight article figures are generated from `publication_results.json`, a compact dataset containing all weekly series used in the plots plus the declared qualification and transition summaries. The renderer verifies the dataset's SHA-256 digest and fails closed if the Python, NumPy, or Matplotlib version differs from the locked environment.

From the repository root:

```text
python -m pip install -r docs/manuscript/figures/requirements.txt
python docs/manuscript/figures/render_figures.py
```

The renderer writes PNG and SVG versions of all eight figures. It does not train or evaluate controllers, run an optimizer, select policies, or perform statistical inference.

Expected environment:

- CPython 3.14.4
- NumPy 2.4.4
- Matplotlib 3.10.8

The compact result file retains cryptographic identifiers for its source populations so that the publication dataset can be reconciled with the research record without publishing operational material.
