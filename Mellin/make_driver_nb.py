"""Generate MellinOscillationsThimbles.nb (driver notebook) from plain-text cells."""
import sys

def esc(s):
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and not s.startswith('\\[', i):
            out.append('\\\\')
        elif ch == '"':
            out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return ''.join(out)

cells = []
def C(style, text): cells.append(f'Cell["{esc(text.strip())}", "{style}"]')

C("Title", "MellinOscillationsOdd: thimbles, Gauss-Hermite and Stokes corrections")
C("Text", """
Computes f(r) = (1/2\\[Pi]i)\\[Integral] r^q ZR(q) dq  (strip -1<Re q<0) and the structure function
D(r) = 2(f(0)-f(r)) = (1/2\\[Pi]i)\\[Integral] r^q (-2 ZR(q)) dq  (strip 0<Re q<2), and the effective index
\\[Alpha](log r) = d log F/d log r, by steepest descent along Lefschetz thimbles with Gauss-Hermite quadrature
(log weight = -t^2 along p(t)), plus the Stokes corrections: residues of the Riemann (7+i\\[Gamma]) and dyadic (15/2+2\\[Pi]ik/log 2)
poles trapped by right-bending thimbles, and the tails of thimbles that terminate on zeros 6+i\\[Gamma] of ZR.
All definitions are in MellinOscillationsThimbles.wl; this notebook runs checks, the scan, and the plots.
""")
C("Section", "Setup")
C("Input", """
SetDirectory[NotebookDirectory[]];
Get["ABCInterpolator.wl"];          (* ABC, \\[CapitalDelta]1, \\[CapitalDelta]2 (needs iqinterp.mx) *)
Get["MellinOscillationsThimbles.wl"];
""")
C("Section", "Checks (old notebook values and the Python reference)")
C("Input", """
{{DF0[-3.], "old NIntegrate DF[-3,0] = 24057.036041758227"},
 {S1q[-0.5], "old S1R[-0.5] = 2.9044428692833426 (S1R was right)"},
 {S2q[-0.5], "Python: 9.06518 (the old S2R gave -7.40176: sign bug in the DF2 term)"},
 {Total[GHNodes[80][[2]]] - Sqrt[Pi], "GH weights: should be ~1e-15"}} // TableForm
""")
C("Input", """
checkXi = {3., 0., -2., -3., -3.5, -4.5, -6.};
checks = Table[With[{r = ComputeMellin[xi, "D"]},
    {xi, r["F"], DirectLine[xi, "D"], r["alpha"], DirectLineAlpha[xi, "D"], r["nRiemann"], r["nDyadic"], r["terminalZero"], r["drift"]}],
   {xi, checkXi}];
Grid[Prepend[checks, {"log r", "D thimble+Stokes", "D direct line", "\\[Alpha] thimble", "\\[Alpha] direct", "trapped Riemann", "trapped dyadic", "ends on zero #", "drift"}],
  Frame -> All, ItemStyle -> {Automatic, {1 -> Bold}}]
""")
C("Text", "Python reference values (Mellin/python, same algorithm, validated against the direct line integral):")
C("Input", "pythonReference = Import[\"results/pythonReference_D.csv\"]; Grid[pythonReference, Frame -> All]")
C("Section", "Scan over log r")
C("Input", """
LaunchKernels[];
DistributeDefinitions[ComputeMellin, DirectLine, DirectLineAlpha, GHNodes];
xiGrid = Range[-8., 8., 0.05];
resD = ParallelTable[ComputeMellin[xi, "D"], {xi, xiGrid}, Method -> "FinestGrained"];
resF = ParallelTable[ComputeMellin[xi, "f"], {xi, xiGrid}, Method -> "FinestGrained"];
""")
C("Section", "Plots")
C("Input", """
regimeBands[res_] := Module[{b, c},
   b = Select[res, #["terminalZero"] =!= None &][[All, "xi"]];
   c = Select[res, #["nRiemann"] + #["nDyadic"] > 0 &][[All, "xi"]];
   {If[b =!= {}, {Opacity[0.12], Orange, Rectangle[{Min[b], -10}, {Max[b], 10}]}, {}],
    If[c =!= {}, {Opacity[0.12], Purple, Rectangle[{Min[c], -10}, {Max[c], 10}]}, {}]}];
pD = ListLogLogPlot[{Exp[#["xi"]], #["F"]} & /@ resD, Joined -> True, PlotStyle -> {Thick, Blue}, Frame -> True,
   FrameLabel -> {"r", "D(r) = 2(f(0) - f(r))"}, ImageSize -> 600, GridLines -> Automatic];
pAlphaD = ListLinePlot[{#["xi"], #["alpha"]} & /@ resD, PlotStyle -> {Thick, Blue}, Frame -> True,
   FrameLabel -> {"log r", "\\[Alpha](log r) = d log D/d log r"}, ImageSize -> 600, GridLines -> Automatic,
   Prolog -> regimeBands[resD], PlotRange -> {All, {-0.05, 2.05}},
   PlotLabel -> "orange: thimble ends on a zero 6+i\\[Gamma]; purple: thimble traps the Riemann/dyadic walls"];
pF = ListLogLogPlot[{Exp[#["xi"]], #["F"]} & /@ resF, Joined -> True, PlotStyle -> {Thick, Darker[Green]}, Frame -> True,
   FrameLabel -> {"r", "f(r)"}, ImageSize -> 600, GridLines -> Automatic];
pAlphaF = ListLinePlot[{#["xi"], #["alpha"]} & /@ resF, PlotStyle -> {Thick, Darker[Green]}, Frame -> True,
   FrameLabel -> {"log r", "\\[Alpha](log r) = d log f/d log r"}, ImageSize -> 600, GridLines -> Automatic, Prolog -> regimeBands[resF]];
Column[{pD, pAlphaD, pF, pAlphaF}]
""")
C("Input", """
(* the oscillating (Stokes) part of alpha: residues of the trapped wall poles and zero tails *)
stokesD = Select[{#["xi"], Abs[#["dalphaStokes"]]} & /@ resD, #[[2]] > 0 &];
ListLogPlot[stokesD, Joined -> True, PlotStyle -> {Thick, Red}, Frame -> True,
  FrameLabel -> {"log r", "|\\[Delta]\\[Alpha] Stokes|"}, ImageSize -> 600, GridLines -> Automatic,
  PlotLabel -> "Stokes (trapped-pole + zero-tail) correction to \\[Alpha] for D(r)"]
""")
C("Input", """
(* Stokes staircases *)
ListStepPlot[{{#["xi"], #["nRiemann"]} & /@ resD, {#["xi"], #["nDyadic"]} & /@ resD,
   {#["xi"], Replace[#["terminalZero"], None -> 0]} & /@ resD},
  Frame -> True, FrameLabel -> {"log r", "count / index"}, ImageSize -> 700,
  PlotLegends -> {"trapped Riemann poles (of " <> ToString[nRiemann] <> ")", "trapped dyadic poles (of " <> ToString[nDyadic] <> ")", "thimble ends on zero 6+i\\[Gamma]_n : n"},
  PlotRange -> {All, {0, 62}}]
""")
C("Input", """
(* thimbles for a few log r, with poles (red), zeros (green) and the walls Re q = 6, 7, 15/2 *)
pick = Flatten[Position[xiGrid, _?(MemberQ[{1., -2., -2.8, -3., -3.5, -4.5, -6.}, Round[#, 0.01]] &)]];
PlotThimbles[resD[[pick]], {-6, 16}, {-2, 50}]
""")
C("Input", """
Export["MellinOsc_D.csv", Prepend[{#["xi"], #["F"], #["alpha"], #["Fthimble"], #["alphaThimble"], #["dFStokes"], #["dalphaStokes"],
     #["nRiemann"], #["nDyadic"], Replace[#["terminalZero"], None -> 0], #["q0"]} & /@ resD,
   {"logr", "D", "alpha", "D_thimble", "alpha_thimble", "dD_Stokes", "dalpha_Stokes", "nRiemann", "nDyadic", "terminalZero", "q0"}]];
Export["MellinOsc_f.csv", Prepend[{#["xi"], #["F"], #["alpha"], #["Fthimble"], #["alphaThimble"], #["dFStokes"], #["dalphaStokes"],
     #["nRiemann"], #["nDyadic"], Replace[#["terminalZero"], None -> 0], #["q0"]} & /@ resF,
   {"logr", "f", "alpha", "f_thimble", "alpha_thimble", "df_Stokes", "dalpha_Stokes", "nRiemann", "nDyadic", "terminalZero", "q0"}]];
""")
C("Section", "Log-periodic (oscillating) part of \\[Alpha]: all Riemann and dyadic wall poles")
C("Text", """
For r < C_min = Min[abcC] (log r < -2.92) D(r) is the convergent sum of its right residues; the only oscillating terms are
r^(7+i\\[Gamma]_n) and r^(15/2+2\\[Pi]ik/log 2). For r > C_max = Max[abcC] (log r > -2.76) D(r) is a convergent series in real powers
of 1/r: no oscillation at all. Where the thimble traps all wall poles (log r < -4.2) the Stokes residue sum equals this wall sum.
""")
C("Input", """
{Log[Min[abcC]], Log[Max[abcC]]}
wallD = Table[With[{r = resD[[i]]}, {r["xi"], WallOscillation[r["xi"], "D", r["F"], r["alpha"]]["dalpha"]}], {i, Length[resD]}];
ListLogPlot[{Select[{#[[1]], Abs[#[[2]]]} & /@ wallD, #[[1]] < Log[Min[abcC]] &], stokesD}, Joined -> {True, False},
  PlotStyle -> {{Thick, Blue}, {Red, PointSize[0.008]}}, Frame -> True, FrameLabel -> {"log r", "|\\[Delta]\\[Alpha]|"},
  PlotLegends -> {"all wall poles (log-periodic part of \\[Alpha])", "Stokes-trapped poles + zero tails"}, ImageSize -> 700]
""")
C("Section", "Your old large-r oscillations (CorrelationOscillation.nb) were a k-cutoff artifact")
C("Text", """
Dv2[r] = (1/\\[Pi]^2)\\[Integral]_0.1^1000 (1-Sin[k r]/(k r)) H(k) dk truncates the k integral at k=0.1 where H(0.1)=0.037 is not small: the endpoint
leaves a term ~ H(0.1) Cos[0.1 r]/r^2 that makes xi2 = r D[Log[Dv2],r] oscillate (period 2\\[Pi]/0.1 in r) and even go negative for log10 r > 1.4.
The exact Mellin D(r) (no cutoff) gives a monotone \\[Alpha] there. Python check: at log10 r = 1.5, 1.75, 2.0 the cut-off xi2 = -3.4e-4, -3.2e-4, -2.3e-4
while the exact \\[Alpha]_D = 1.77e-3, 9.9e-4, 5.6e-4.
""")
C("Input", """
alphaDInterp = Interpolation[{#["xi"], #["alpha"]} & /@ resD];
Plot[alphaDInterp[t Log[10.]], {t, 1, 2.5}, PlotStyle -> {Thick, Blue}, Frame -> True, PlotRange -> All,
  FrameLabel -> {"log10 r", "\\[Alpha]_D = r D[Log[D(r)], r]"}, PlotLabel -> "exact index at large r (compare X1 in CorrelationOscillation.nb)", ImageSize -> 600]
""")
C("Section", "Max Planck wind tunnel (E_Kohler): measured index vs theory")
C("Text", """
Each file E_Kohler/Re_<Re_lambda>_Eps_<eps>.csv has columns r/eta, S_2, S_3. The measured index is d Log S_2/d Log r (centered differences).
As in CorrelationOscillation.nb the large-r tail (index < 0.355, beyond the inertial plateau) is fitted by a shift s: \\[Alpha]_exp(Log(r/\\[Eta])) = \\[Alpha]_D(Log(r/\\[Eta]) - s).
The theory has no inertial plateau, so only the tail is compared. The fitted s should grow like Log(L/\\[Eta]) ~ (3/2) Log Re_lambda.
""")
C("Input", """
mpiFiles = SortBy[FileNames["Re_*.csv", "E_Kohler"], ToExpression[First[StringCases[FileBaseName[#], "Re_" ~~ x : NumberString ~~ "_" :> x]]] &];
alphaExp[f_] := Module[{d = Rest[Import[f, "CSV"]], lr, lS},
   lr = Log[N[d[[All, 2]]]]; lS = Log[N[d[[All, 3]]]];
   Transpose[{lr, Join[{(lS[[2]] - lS[[1]])/(lr[[2]] - lr[[1]])}, (lS[[3 ;;]] - lS[[;; -3]])/(lr[[3 ;;]] - lr[[;; -3]]),
      {(lS[[-1]] - lS[[-2]])/(lr[[-1]] - lr[[-2]])}]}]];
fitTail[f_] := Module[{a = alphaExp[f], i0, tail, s, sol},
   i0 = First[FirstPosition[Transpose[{a[[All, 2]], a[[All, 1]]}], {x_ /; x < 0.355, y_ /; y > Mean[a[[All, 1]]]}]];
   tail = a[[i0 ;;]];
   sol = FindMinimum[Total[(tail[[All, 2]] - alphaDInterp[tail[[All, 1]] - s])^2], {s, 10., 5., 15.}];
   <|"file" -> FileBaseName[f], "Re" -> ToExpression[First[StringCases[FileBaseName[f], "Re_" ~~ x : NumberString ~~ "_" :> x]]],
     "shift" -> (s /. sol[[2]]), "rms" -> Sqrt[sol[[1]]/Length[tail]], "n" -> Length[tail], "alpha" -> a|>];
fits = fitTail /@ mpiFiles;
Grid[Prepend[{#["Re"], #["shift"], #["shift"]/Log[10.], #["rms"], #["n"]} & /@ fits, {"Re_lambda", "shift s (ln)", "s (log10)", "rms", "points"}], Frame -> All]
""")
C("Input", """
reCols = ColorData["BlueGreenYellow"] /@ Subdivide[0, 1, Length[fits] - 1];
Show[ListPlot[Table[{#[[1]] - fits[[i]]["shift"], #[[2]]} & /@ fits[[i]]["alpha"], {i, Length[fits]}],
   PlotStyle -> reCols, PlotMarkers -> {Automatic, 5}, PlotLegends -> ("Re_\\[Lambda]=" <> ToString[Round[#["Re"]]] & /@ fits)],
  Plot[alphaDInterp[x], {x, -8, 3}, PlotStyle -> {Black, Thick}], Frame -> True, PlotRange -> {{-8, 3}, {-0.15, 2.05}},
  FrameLabel -> {"Log(r/\\[Eta]) - s", "\\[Alpha] = d Log S_2/d Log r"}, ImageSize -> 800, GridLines -> Automatic]
LinearModelFit[{Log[#["Re"]], #["shift"]} & /@ fits, x, x]["BestFitParameters"]   (* slope ~ 1.35 in Python; K41: 1.5 *)
""")
C("Section", "Fixes needed in the original MellinOscillationsOdd.nb cells (for reference)")
C("Text", """
1. S1Interp[q] = -df1approx[q] + S1Analytic[q]   (was +df1approx: d/dq Log DF[-1-q,0] = -DF1/DF0).
2. S2R[q] = +DF[-1-q,2]/DF[-1-q,0] - (DF[-1-q,1]/DF[-1-q,0])^2 + S2Analytic[q]   (was -DF2/DF0). With the bug S2R[-0.5] = -7.40 < 0, v0 becomes real and the path runs along the real axis until S1Interp+xi = 0 (the Infinite expression messages).
3. p0Values must exclude q0 = 0 (pole of Csc).
4. Walls: Riemann poles at 7 + I gamma_n (not 9 or -8), dyadic poles at 15/2 + 2 Pi I k/Log[2] (not -17/2); zeros at 6 + I gamma_n.
5. CountTrappedPoles: rightward ray (reCross > wall), and trapped residues enter with a MINUS sign: F = F_thimble - 2 Re Sum Res.
6. S0 = Log[ZR[q0]] + xi q0 and xi = -S1R[q0] (not the old p-problem S[p0], S1[p0]).
7. DF off the [-1,0]x[0,10] interpolation grid: df0values kept only Re Log DF and df1approx falls back to a constant; here DF is evaluated exactly (Gauss-Legendre in \\[CapitalDelta]) everywhere.
""")

nb = 'Notebook[{\n' + ',\n'.join(cells) + '\n}, WindowSize -> {1100, 900}]\n'
open(sys.argv[1] if len(sys.argv) > 1 else 'MellinOscillationsThimbles.nb', 'w').write(
    '(* Content-type: application/vnd.wolfram.mathematica *)\n\n' + nb)
