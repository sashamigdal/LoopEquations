"""Paths, the Max Planck file reader and the plotting style shared by all scripts.

Layout (relative to this folder):
    ../results/       tables committed with the code (JSON/CSV) and the figures of the README
    ../paper/figs/    the figures of the paper (written by paper_figures.py)
    cache/            large intermediate arrays (.npy, full MPI fit records); not committed
    $MPI_DATA_DIR     the Max Planck files Re_<Re_lambda>_Eps_<eps>.csv  (default ../E_Kohler; not distributed)
"""
import glob, json, os, re
import numpy as np

PY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PY)
RESULTS = os.path.join(ROOT, 'results')
PAPER_FIGS = os.path.join(ROOT, 'paper', 'figs')
CACHE = os.path.join(PY, 'cache')
MPI_DIR = os.environ.get('MPI_DATA_DIR', os.path.join(ROOT, 'E_Kohler'))
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def res(name):
    return os.path.join(RESULTS, name)


def cache(name):
    return os.path.join(CACHE, name)


def save_json(obj, path, indent=None):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=indent)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------- Max Planck data (Kuechler, Bewley, Bodenschatz) ----------------
num = lambda pat, f: float(re.search(pat, os.path.basename(f)).group(1).rstrip('.'))


def mpi_files():
    """the 11 runs, sorted by Re_lambda; columns of each CSV: index, r/eta, S_2 (m^2/s^2), S_3"""
    files = sorted(glob.glob(os.path.join(MPI_DIR, 'Re_*.csv')), key=lambda f: num(r'Re_([0-9.]+)_', f))
    if not files:
        raise SystemExit('Max Planck data not found in %s (set MPI_DATA_DIR). They are available from the authors of '
                         'Kuechler, Bewley, Bodenschatz, PRL 131, 024001 (2023).' % MPI_DIR)
    return files


def have_mpi():
    return bool(glob.glob(os.path.join(MPI_DIR, 'Re_*.csv')))


def mpi_index(f):
    """log(r/eta), measured index alpha = d log S2/d log r (centred differences), start of the fitted tail alpha < ALPHA_FIT"""
    dd = np.genfromtxt(f, delimiter=',', skip_header=1); r, S2 = dd[:, 1], dd[:, 2]
    lr, lS = np.log(r), np.log(S2)
    a = np.empty_like(lr); a[1:-1] = (lS[2:]-lS[:-2])/(lr[2:]-lr[:-2])
    a[0] = (lS[1]-lS[0])/(lr[1]-lr[0]); a[-1] = (lS[-1]-lS[-2])/(lr[-1]-lr[-2])
    i0 = int(np.argmax((a < ALPHA_FIT) & (lr > lr.mean())))
    return lr, a, i0


def strip(records, keys=('lr', 'a', 'i0', 'binned')):
    """drop the per-run measured arrays before a table is committed (the data are not ours to distribute)"""
    if isinstance(records, list):
        return [{k: v for k, v in r.items() if k not in keys} for r in records]
    return {k: v for k, v in records.items() if k not in keys}


# ---------------- plotting style ----------------
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
C1, C2, C3, C4, C5, C6, C7 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b', '#0a2a55']
LOGCMIN, LOGCMAX = -2.9153, -2.7635   # log of min/max C(Delta)
ALPHA_FIT = 0.40    # the theory is fitted to the tail alpha < ALPHA_FIT (per run and after the Re -> infinity extrapolation)
ALPHA_ATTR = 0.50   # attractor region alpha <= ALPHA_ATTR: the index scales with the run's own large-scale length (attractor_region.py)


def style(titles=True):
    """matplotlib in Agg mode with the house style; titles=False suppresses in-figure titles (paper figures)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                         'grid.linewidth': 0.6, 'lines.linewidth': 2, 'font.size': 10.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})
    if not titles:
        import matplotlib.axes, matplotlib.figure
        matplotlib.axes.Axes.set_title = lambda self, *a, **k: None
        matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
    return plt
