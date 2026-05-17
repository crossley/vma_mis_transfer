# vma_mis_transfer

Visuomotor adaptation experiment code and trial-design tools for estimating
different sources of reaching variability.

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
