<div align="center">
   <h1 align="center">Loop Equations</h1>
   <img alt="circular loop with beads, with LE written inside for loop equations" src="https://raw.githubusercontent.com/sashamigdal/LoopEquations/master/logo.jpg" width="30%" />
</div>

This collection of Mathematica notebooks and Python scripts supports the enclosed paper  Notebooks/Microscopic_theory_of_Decaying_Turbulence_MDPI.pdf

Mathematica notebooks verify analytic solutions, and the Python code in VorticityCorrelation.py simulates the random complex curve with N   up to 2*10^8 points of a polygon

and T= 2*10^5 samples.

The formulas are explained in the text of the paper.

The code in VorticityCorrelation.py has two regimes:

1. statistics collection ( script 'run.sh" for a cluster  and "run_local.sh" for a laptop/desktop) Output is stored in "FData.T.C.np" files
   (T is the number of samples, and C is the number of a node that created these samples on a cluster in a parallel run.
2. The second regime (script plot.sh and plot_local.sh, automatically launched in the "submit.sh" script after the statistics collection) is the data processing part\\
   This processing starts with pooling all the "FData....C.np" files for C =1,... Num_nodes into a single data file. Then this data file is used to plot various distributions


# building C++ library
1. `cd CPP`

2. activate G++ 7 somehow

3. `cmake .`

4. `make` 
