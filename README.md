# vma_mis_transfer

Visuomotor adaptation experiment code and trial-design tools for estimating
different sources of reaching variability.

## Requirements

Python 3.11 is required (developed and tested on 3.11.8). The versions below
are what was used during development; newer patch releases should work.

### Experiment runner (`code/`)

Install pinned dependencies from the requirements file:

```bash
pip install -r code/requirements.txt
```

If you are using the Liberty tracker (`use_liberty = True` in `run_exp.py`),
also install pyserial:

```bash
pip install pyserial==3.5
```

| Package   | Tested version | Notes                              |
|-----------|----------------|------------------------------------|
| numpy     | 1.26.4         |                                    |
| pandas    | 2.2.1          |                                    |
| matplotlib| 3.8.3          |                                    |
| pygame    | 2.5.2          |                                    |
| pyserial  | 3.5            | Only needed when `use_liberty = True` |

### Noise-model design tools (`code_noise_model/`)

Install the package in editable mode from the `code_noise_model/` directory:

```bash
pip install -e code_noise_model/
```

This installs all required dependencies at their pinned versions (numpy,
pandas, scipy, matplotlib, scikit-learn, statsmodels). Additionally install
seaborn for the plotting scripts:

```bash
pip install seaborn==0.13.2
```

| Package      | Tested version |
|--------------|----------------|
| numpy        | 1.26.4         |
| pandas       | 2.2.1          |
| scipy        | 1.16.0         |
| matplotlib   | 3.8.3          |
| scikit-learn | 1.5.1          |
| statsmodels  | 0.14.5         |
| seaborn      | 0.13.2         |

## Contents

- `code/run_exp.py`: Pygame experiment runner. It supports mouse input for
  testing and Liberty tracking for the lab setup. The current inline trial
  structure uses the `nf_burst_after_clamp` diagnostic design followed by
  adaptation and generalization phases.
- `code_noise_model/`: Python package used to simulate candidate trial
  sequences and evaluate which designs best recover output-only versus
  process-noise models.
- `consent/`: consent/PICF materials.

## Running the Experiment

From the `code/` directory:

```bash
python run_exp.py
```

The script prompts for subject number and day number, then writes data to
`data/sub_<subject>_day_<day>_data.csv` and movement samples to
`data/sub_<subject>_day_<day>_data_move.csv`. Existing files with the same
subject/day are not overwritten.

Set `use_liberty = True` in `code/run_exp.py` to use the Liberty tracking
system. Leave it as `False` for mouse input.

## Noise-Model Design Tools

From the `code_noise_model/` directory:

```bash
python examples/run_design_shortlist.py
```

The package simulates and fits implicit adaptation noise models to evaluate
which trial sequences best identify process-noise structure. Generated
outputs and reports are written under `code_noise_model/outputs/` and
`code_noise_model/reports/`, which are ignored by git.
