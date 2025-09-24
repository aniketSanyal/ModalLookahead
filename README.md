# Modal Lookahead (Paper Code)


## Features
- Problems: bilinear and strongly-convex–strongly-concave (SC-SC) quadratics.
- Optimizers: Fixed Lookahead baseline; Modal Lookahead with k_phase/k_amp rule.
- Controllers (spectral estimation):
  - mode=grad_bilinear: first-order only, estimates dominant sigma via power on A^T A using grad oracles.
  - mode=hvp: second-order (HVP) oracle, complex power iteration on G = I - gamma J (no matrix assembly).
- Plotting: styled figures suitable for papers (PNG/PDF/SVG).

## Install
```bash
pip install -r requirements.txt
```

## Layout
```
src/modal_lookahead/
  problems/
    bilinear.py         # x^T A y game
    scsc.py             # SC-SC quadratic (mu_x I, mu_y I, coupled by A)
  optim/
    lookahead.py        # FixedLookahead, ModalLookahead (k_phase/k_amp, two backends)
  controllers/
    spectral_power.py   # power iters: A^T A (first-order) and complex PI on G (HVP)
  utils/
    linalg.py           # shared linear algebra (Krylov / power)
  plotting/
    style.py, plots.py  # paper-ready plotting
  runners/
    run_scsc_random.py  # random SC-SC generator & comparison (Modal vs Fixed)
scripts/
  run_scsc_random.sh    # convenience wrapper
```

## Quick Start: SC-SC random problems
Run Modal Lookahead vs Fixed Lookahead on a random SC-SC game with slightly more rotation than potential:
```bash
# Gradient-only backend (no second-order oracle):
export PYTHONPATH=src
python -m modal_lookahead.runners.run_scsc_random   --d 120 --T 300 --gamma 0.1   --mu_x 0.18 --mu_y 0.18   --sigma_min 0.25 --sigma_max 0.45   --alpha 0.5 --mode grad_bilinear   --outfile figures/scsc_modal_vs_fixed_grad

# HVP backend (requires problem.hvp):
python -m modal_lookahead.runners.run_scsc_random   --d 120 --T 300 --gamma 0.1   --mu_x 0.18 --mu_y 0.18   --sigma_min 0.25 --sigma_max 0.45   --alpha 0.5 --mode hvp   --outfile figures/scsc_modal_vs_fixed_hvp
```

Artifacts are saved at --outfile.{png,pdf,svg}.

## API Hints
To add a new problem, implement:
```python
class MyProblem:
    d: int; gamma: float
    def grad_x(self, y): ...
    def grad_y(self, x): ...
    def step_gda(self, x, y): ...
    def distance(self, x, y): ...
    # Optional for hvp backend:
    def hvp(self, v): ...  # returns J v
```
Modal Lookahead then works out-of-the-box with mode="grad_bilinear" or mode="hvp".

## License
MIT
