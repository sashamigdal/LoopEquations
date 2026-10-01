#!/usr/bin/env bash
# Reproduces the numbers and the computed figures of
#   A. Migdal, "Comparing the Decaying Turbulence Theory with Wind Tunnel Experiments".
# Usage (from any directory):   bash run_all.sh            paper pipeline (about 30 min on 4 cores)
#                               bash run_all.sh --readme   also the README figures and side checks
# The Max Planck part runs only when the data files Re_<Re>_Eps_<eps>.csv are found in $MPI_DATA_DIR
# (default: ../E_Kohler). They are available from the authors of Kuechler, Bewley, Bodenschatz, PRL 131, 024001 (2023).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
export MPI_DATA_DIR=${MPI_DATA_DIR:-$(cd .. && pwd)/E_Kohler}
step() { echo; echo "=== $*"; "$PY" "$@"; }

# 0. The table abc_cheb48.json of A, B, C(Delta) (Chebyshev nodes) is shipped. To rebuild it from the
#    definitions of ABCInterpolator.wl (about 1 h):   python3 build_abc_table.py 48

# 1. Theory (Sec. II of the paper)
step direct_D.py            # D(rho), alpha_D by the direct contour integral on Re q = 1   -> cache/lineD_c1.npy, cache/direct_D.npy
step scan.py D -8 8 0.05    # saddle, thimble, Gauss-Hermite, Stokes terms for D          -> ../results/scan_D.json
step scan.py f -8 8 0.05    # the same for the correlation function G                     -> ../results/scan_f.json
step hseries.py             # entire series of H(kappa), (pi/2) H(0) = a_1                  -> cache/hcoef.npy
step oscill.py              # log-periodic (wall-pole) part of D and alpha_D                -> cache/oscill.npy
step validate.py            # derivative/residue checks, thimbles vs direct integral       -> ../results/pythonReference_D.csv
step spectrum_osc.py        # the same for the energy spectrum; checks Z(q) against M(p)    -> cache/spectrum_osc.npy
step lineH.py               # H(kappa), dH/dlog kappa on the line Re p = -1.5 (reference)     -> cache/lineH_m15.npy
step stokes_events.py       # Stokes events of H and D: bisection, secondary saddles, checks -> ../results/stokes_events.json
step stokes_berry.py        # Berry-smoothed trapped-pole part of the two indices            -> cache/stokes_berry.npz

# 2. Max Planck comparison (Sec. III)
if ls "$MPI_DATA_DIR"/Re_*.csv >/dev/null 2>&1; then
  step finite_box.py        # per run: shift s (infinite system) and (s, kappa_min) (finite width) -> ../results/mpi_fit_box.json
  step physical_units.py    # u', nu, eta, l_D, implied width W in metres                     -> ../results/mpi_physical_units.json
  step finite_box_W.py      # one physical width W for all runs                               -> ../results/mpi_fit_width.json
  step finite_box_W2.py     # one W for the Re_lambda <= 2398 runs: W = 2.34 m                -> ../results/mpi_fit_width_sub.json
  step extrapolate_Re.py    # integral scale L per run; Re^-1 and Re^-1/2 extrapolations       -> ../results/mpi_extrapolate_Re_inf.json
  step extrap_invRe.py      # adopted 1/Re_lambda extrapolation and theory fit (Table I)      -> ../results/mpi_extrap_invRe.json
  step attractor_region.py  # fixed-alpha cross-sections: turbulent attractor vs stochastization stage -> ../results/mpi_attractor_region.json
  step fit_robustness.py    # the fit vs the tail cut, leave-one-out and bootstrap over runs  -> ../results/mpi_fit_robustness.json
else
  echo; echo "=== Max Planck data not found in $MPI_DATA_DIR: Sec. III fits skipped (the committed tables in ../results are used)"
fi
if ls "${MATSUZAWA_DATA_DIR:-$(cd .. && pwd)/Matsuzawa}"/*Fig3A_dissipation_rate.h5 >/dev/null 2>&1; then
  step matsuzawa_enstrophy.py  # enstrophy decay of the Matsuzawa et al. blob: 9/4 vs 11/5     -> ../results/matsuzawa_enstrophy.json
fi
step even_vs_odd.py         # odd vs even ensemble fitted to the Re -> infinity tail (uses ../results/mpi_extrap_invRe.json)
step tail_check.py          # Sec. III D: index below r = L vs theory, Re trend at fixed r/L, finite width at small rho

# 3. Figures of the paper -> ../paper/figs
step paper_figures.py

if [[ "${1:-}" == "--readme" ]]; then
  step cutoff_test.py       # the large-r oscillations of CorrelationOscillation.nb come from the cutoff kappa > 0.1
  if ls "$MPI_DATA_DIR"/Re_*.csv >/dev/null 2>&1; then
    step mpi_fit_tail.py; step finite_box_global.py; step fig_box.py; step fig_extrap.py; step fig_invRe.py
  fi
  step make_figures.py      # README figures 1-8 -> ../results
fi
echo; echo "done"
