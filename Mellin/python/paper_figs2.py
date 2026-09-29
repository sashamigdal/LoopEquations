"""Re-run the figure scripts for the paper without in-figure titles."""
import sys, runpy, matplotlib
matplotlib.use('Agg')
import matplotlib.axes, matplotlib.figure
matplotlib.axes.Axes.set_title = lambda self, *a, **k: None
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
out = sys.argv[1]
for script in ['make_figures_paper.py', 'fig_box_paperlabels.py', 'fig_invRe.py']:
    sys.argv = [script, out]; runpy.run_path(script, run_name='__main__')
