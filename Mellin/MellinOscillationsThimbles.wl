(* ::Package:: *)

(* ::Title:: *)
(*MellinOscillationsThimbles.wl*)


(* ::Text:: *)
(*Steepest descent (Lefschetz thimbles) + Gauss-Hermite + Stokes corrections for*)
(*   f(r) = (1/2 Pi I) Integrate[ r^q ZR[q], {q, c - I Infinity, c + I Infinity}],        -1 < c < 0,*)
(*   D(r) = 2 (f(0) - f(r)) = (1/2 Pi I) Integrate[ r^q (-2 ZR[q]), ...],               0 < c < 2,*)
(*   ZR[q] = -(64 Sqrt[2] Pi Csc[Pi q/2] DF[-1-q,0] Zeta[13/2-q]) / ((128 Sqrt[2]-2^q)(1+q)(2q-15)(2q-5) Zeta[15/2-q]),*)
(*and the effective index alpha(Log r) = r D[Log[F[r]], r] = d Log F / d xi,  xi = Log[r],  F = f or D.*)
(**)
(*Thimble: W(q) = S(q) + xi q with e^S = pref ZR (pref = 1 for f, -2 for D).  Saddle q0 real in the strip, W'(q0) = 0.*)
(*   W(q(t)) = W0 - t^2,   q'(t) = -2 t / W'(q),   q(-t) = Conjugate[q(t)].*)
(*   F = (E^W0/Pi) ( Sum_{t_i>0} w_i Im[g(t_i)] ),   g(t) = E^(W(q(t)) - W0 + t^2) q'(t)   (Gauss-Hermite, weight E^(-t^2)).*)
(*   On an exact thimble E^(W - W0 + t^2) = 1; keeping the factor makes the sum exact by Cauchy even if the*)
(*   numerical path drifts (it is also the drift diagnostic).*)
(*   dF/dxi uses the same nodes with an extra factor q, so alpha needs no numerical differentiation.*)
(*Stokes corrections (the new walls are on the RIGHT: Riemann poles 7 + I gamma_n, dyadic poles 15/2 + 2 Pi I k/Log[2]):*)
(*   F = F_thimble - 2 Re[ Sum over poles trapped between Re q = c and the upper thimble of Res ]*)
(*   (trapped <=> odd number of crossings of the RIGHTWARD ray Im q = Im pole, Re q > wall),*)
(*   plus, when the thimble terminates on a zero 6 + I gamma_n of ZR, the tail up Re q = 6.*)
(**)
(*Requires ABCInterpolator.wl to be loaded first (defines ABC, \[CapitalDelta]1, \[CapitalDelta]2).*)
(*Validated against an independent Python implementation (Mellin/python): thimble+GH agrees with the direct*)
(*vertical-line integral to 1e-10 .. 1e-13 for xi in [-8, 8].*)


(* ::Section:: *)
(*1. Fast exact DF(p): fixed Gauss-Legendre grid in \[CapitalDelta] (replaces NIntegrate; valid for any complex p)*)


Needs["NumericalDifferentialEquationAnalysis`"];

nGL = 160;
glxw = GaussianQuadratureWeights[nGL, \[CapitalDelta]1, \[CapitalDelta]2, 30];
glx = N[glxw[[All, 1]]]; glw = N[glxw[[All, 2]]];
abcTab = ABC /@ glx;                       (* {A, B, C} at the nodes; ABC is smooth on [\[CapitalDelta]1, \[CapitalDelta]2] *)
abcA = abcTab[[All, 1]]; abcB = abcTab[[All, 2]]; abcC = abcTab[[All, 3]];
abcLC = Log[abcC];
dfW = 20 glw (1 - glx);

(* DFk[p] = D[DF[p,0], {p,k}] = Integrate[(1-x) ZIk[p, ABC[x]], {x, \[CapitalDelta]1, \[CapitalDelta]2}] *)
DF0[p_?NumericQ] := Total[dfW Exp[(p - 1) abcLC] (abcA abcC - abcB p)];
DF1[p_?NumericQ] := Total[dfW Exp[(p - 1) abcLC] (-abcB + (abcA abcC - abcB p) abcLC)];
DF2[p_?NumericQ] := Total[dfW Exp[(p - 1) abcLC] abcLC (-2 abcB + (abcA abcC - abcB p) abcLC)];


(* ::Section:: *)
(*2. Integrand, exact log-derivatives (sign-corrected S1R / S2R)*)


KK = 128 Sqrt[2.];
pref["f"] = 1.; pref["D"] = -2.;
strip["f"] = {-1., 0.}; strip["D"] = {0., 2.};

(* analytic factor ZR[q]/DF[-1-q,0] *)
RAn[q_?NumericQ] := -(64 Sqrt[2.] Pi Csc[Pi q/2] Zeta[13/2 - q])/((KK - 2^q) (1 + q) (2 q - 15) (2 q - 5) Zeta[15/2 - q]);
ZRq[q_?NumericQ] := RAn[q] DF0[-1 - q];
(* Log of the integrand (branch irrelevant: only exponentiated differences are used) *)
logZ[q_?NumericQ, kind_String] := Log[pref[kind] RAn[q]] + Log[DF0[-1 - q]];

(* S'(q) = d/dq Log[ZR[q]].  Note d/dq Log[DF[-1-q,0]] = -DF1/DF0  (the old S1Interp had +df1approx: sign bug) *)
S1q[q_?NumericQ] := -Pi/2 Cot[Pi q/2] - Zeta'[13/2 - q]/Zeta[13/2 - q] + Zeta'[15/2 - q]/Zeta[15/2 - q] +
    2^q Log[2.]/(KK - 2^q) - 1/(1 + q) - 2/(2 q - 15) - 2/(2 q - 5) - DF1[-1 - q]/DF0[-1 - q];

(* S''(q).  Note d^2/dq^2 Log[DF[-1-q,0]] = +DF2/DF0 - (DF1/DF0)^2  (the old S2R had -DF2/DF0: sign bug) *)
S2q[q_?NumericQ] := Module[{p = -1 - q, d0, d1, d2, lz2},
    lz2[s_] := Zeta''[s]/Zeta[s] - (Zeta'[s]/Zeta[s])^2;
    d0 = DF0[p]; d1 = DF1[p]; d2 = DF2[p];
    (Pi/2)^2 Csc[Pi q/2]^2 + lz2[13/2 - q] - lz2[15/2 - q] + 2^q Log[2.]^2 KK/(KK - 2^q)^2 +
     1/(1 + q)^2 + 4/(2 q - 15)^2 + 4/(2 q - 5)^2 + d2/d0 - (d1/d0)^2];

(* saddle point: -S'(q0) = xi, q0 real in the strip (S'' > 0 there, so q0 is unique) *)
Saddle[xi_?NumericQ, kind_String] := Module[{a, b, q},
    {a, b} = strip[kind] + {10.^-7, -10.^-7};
    q /. FindRoot[Re[S1q[q]] + xi, {q, a, b}, Method -> "Brent", AccuracyGoal -> 14, PrecisionGoal -> 14]];


(* ::Section:: *)
(*3. Thimble: NDSolve of q'(t) = -2 t/(S'(q) + xi)*)


BuildThimble[xi_?NumericQ, kind_String, tMax_: 12.3, eps_: 0.001] := Module[{q0, s2, s3, v0, c2, sol, p, t},
    q0 = Saddle[xi, kind];
    s2 = Re[S2q[q0]];
    s3 = (Re[S2q[q0 + 10.^-4]] - Re[S2q[q0 - 10.^-4]])/(2 10.^-4);
    v0 = I Sqrt[2/s2];                     (* steepest-descent direction at the saddle *)
    c2 = s3/(3 s2^2);                      (* q = q0 + v0 t + c2 t^2 + O(t^3) *)
    sol = Quiet[NDSolveValue[{p'[t] == -2 t/(S1q[p[t]] + xi), p[eps] == q0 + v0 eps + c2 eps^2,
         WhenEvent[Abs[p[t]] > 400, "StopIntegration"]}, p, {t, eps, tMax},
        Method -> {"ExplicitRungeKutta", "DifferenceOrder" -> 8}, AccuracyGoal -> 12, PrecisionGoal -> 11,
        MaxSteps -> 10^6], {NDSolveValue::precw, NDSolveValue::mxst}];
    <|"xi" -> xi, "kind" -> kind, "q0" -> q0, "S2" -> s2, "v0" -> v0, "c2" -> c2, "eps" -> eps, "p" -> sol|>];

ThimbleEnd[th_Association] := th["p"]["Domain"][[1, 2]];
ThimblePoint[th_Association, t_?NumericQ] := If[t < th["eps"], th["q0"] + th["v0"] t + th["c2"] t^2, th["p"][t]];


(* ::Section:: *)
(*4. Gauss-Hermite along the thimble (weight E^(-t^2))*)


(* Golub-Welsch nodes/weights for Integrate[E^(-t^2) g(t), {t, -Infinity, Infinity}] *)
GHNodes[n_Integer] := GHNodes[n] = Module[{J, vals, vecs, ord},
    J = N[SparseArray[{Band[{1, 2}] -> Sqrt[Range[n - 1]/2], Band[{2, 1}] -> Sqrt[Range[n - 1]/2]}, {n, n}]];
    {vals, vecs} = Eigensystem[Normal[J]];
    ord = Ordering[vals];
    {vals[[ord]], Sqrt[Pi] (vecs[[ord, 1]])^2}];

(* returns F_thimble and dF_thimble/dxi; the lower branch is the complex conjugate, hence (1/Pi) Im *)
ThimbleGHSum[th_Association, nGH_Integer: 80] := Module[
    {xi = th["xi"], kind = th["kind"], q0 = th["q0"], v0 = th["v0"], xs, ws, w0, sel, ts, wts, qs, dq, W0, dW, g, i0, i1},
    {xs, ws} = GHNodes[nGH];
    w0 = If[OddQ[nGH], ws[[(nGH + 1)/2]], 0.];
    sel = Flatten[Position[xs, _?(# > 10.^-12 &)]];
    ts = xs[[sel]]; wts = ws[[sel]];
    sel = Flatten[Position[ts, _?(# <= ThimbleEnd[th] &)]];      (* never extrapolate the path *)
    ts = ts[[sel]]; wts = wts[[sel]];
    qs = ThimblePoint[th, #] & /@ ts;
    dq = -2 ts/((S1q /@ qs) + xi);
    W0 = Re[logZ[q0, kind]] + xi q0;
    dW = (logZ[#, kind] & /@ qs) + xi qs - W0 + ts^2;           (* = 0 mod 2 Pi I on an exact thimble *)
    g = Exp[dW] dq;
    i0 = Exp[W0]/Pi (Total[wts Im[g]] + w0 Im[v0]/2);
    i1 = Exp[W0]/Pi (Total[wts Im[qs g]] + w0 q0 Im[v0]/2);
    <|"I" -> i0, "I1" -> i1, "W0" -> W0, "missingNodes" -> Count[xs, _?(# > 10.^-12 &)] - Length[ts],
      (* weighted deviation of the numerical path from an exact thimble, relative to the sum (0 on an exact thimble) *)
      "drift" -> If[ts === {}, 0., Max[wts Abs[g - dq]]/Max[Total[wts Abs[dq]], 10.^-300]]|>];


(* ::Section:: *)
(*5. Stokes phenomenon: trapped poles (rightward rays), residues, zero-terminated thimbles*)


nRiemann = 60; nDyadic = 40;
gammaList = N[Im[ZetaZero[Range[nRiemann]]]];
dyadList = N[2 Pi Range[nDyadic]/Log[2]];

DensePath[th_Association, n_: 4000] := Prepend[th["p"] /@ Subdivide[th["eps"], ThimbleEnd[th], n], th["q0"]];

(* poles wall + I h enclosed between Re q = c and the upper thimble: odd number of crossings of the
   rightward ray {Im q = h, Re q > wall}.  The unfinished path is continued from its end point along its
   final direction (it goes to infinity with a fixed slope); a thimble that terminates on a zero 6 + I gamma_n
   is continued straight up Re q = 6 instead (left of both walls: no extra crossings, continueEnd -> False).
   Returns the list of enclosed indices. *)
CountTrappedPoles[th_Association, heights_List, wall_?NumericQ, continueEnd_: True] := Module[{P, x, y, d, out = {}, h, s, idx, xc, cnt},
    P = DensePath[th]; x = Re[P]; y = Im[P]; d = P[[-1]] - P[[-200]];
    Do[
     h = heights[[n]];
     s = (Most[y] - h) (Rest[y] - h);
     idx = Select[Flatten[Position[s, _?(# <= 0 &)]], y[[# + 1]] != y[[#]] &];
     xc = (x[[#]] + (h - y[[#]]) (x[[# + 1]] - x[[#]])/(y[[# + 1]] - y[[#]])) & /@ idx;
     cnt = Count[xc, _?(# > wall &)];
     If[TrueQ[continueEnd] && h > y[[-1]] && Im[d] > 0 && x[[-1]] + (h - y[[-1]]) Re[d]/Im[d] > wall, cnt++];
     If[OddQ[cnt], AppendTo[out, n]],
     {n, Length[heights]}];
    out];

(* residues of r^q pref ZR[q]   (checked against circle integrals to 1e-12) *)
RiemannResidue[qn_, xi_, kind_String] := pref[kind] Exp[xi qn] (-64 Sqrt[2.] Pi Csc[Pi qn/2] Zeta[13/2 - qn] DF0[-1 - qn])/
    ((KK - 2^qn) (1 + qn) (2 qn - 15) (2 qn - 5) (-Zeta'[15/2 - qn]));            (* q_n = 7 + I gamma_n *)
DyadicResidue[qk_, xi_, kind_String] := pref[kind] Exp[xi qk] (-64 Sqrt[2.] Pi Csc[Pi qk/2] Zeta[13/2 - qk] DF0[-1 - qk])/
    ((-KK Log[2.]) (1 + qk) (2 qk - 15) (2 qk - 5) Zeta[15/2 - qk]);               (* q_k = 15/2 + 2 Pi I k/Log[2] *)

(* thimbles with -4 < xi < -3 (D) end ON a zero 6 + I gamma_n of ZR (new Stokes staircase);
   the contour then continues from that zero up Re q = 6: tail = (1/Pi) Re Integrate[E^W(6 + I y), {y, gamma_n, Infinity}] *)
TerminalZero[th_Association] := Module[{e = th["p"][ThimbleEnd[th]], k},
    k = FirstPosition[Abs[e - (6 + I gammaList)], _?(# < 10.^-3 &)];
    If[MissingQ[k], None, First[k]]];
ZeroTail[n_Integer, xi_?NumericQ, kind_String] := Module[{g0 = gammaList[[n]], y},
    {1/Pi NIntegrate[Re[Exp[logZ[6 + I y, kind] + xi (6 + I y)]], {y, g0, g0 + 5, g0 + 20, g0 + 60}],
     1/Pi NIntegrate[Re[(6 + I y) Exp[logZ[6 + I y, kind] + xi (6 + I y)]], {y, g0, g0 + 5, g0 + 20, g0 + 60}]}];


(* ::Section:: *)
(*6. Driver: F(xi), alpha(xi), Stokes bookkeeping*)


ComputeMellin[xi_?NumericQ, kind_String, nGH_Integer: 80] := Module[{th, gh, iR, iD, qR, qD, resR, resD, z, tail = {0., 0.}, dI, dI1},
    th = BuildThimble[xi, kind];
    gh = ThimbleGHSum[th, nGH];
    z = TerminalZero[th];
    iR = CountTrappedPoles[th, gammaList, 7., z === None];
    iD = CountTrappedPoles[th, dyadList, 7.5, z === None];
    qR = 7 + I gammaList[[iR]]; qD = 15/2 + I dyadList[[iD]];
    resR = RiemannResidue[#, xi, kind] & /@ qR;
    resD = DyadicResidue[#, xi, kind] & /@ qD;
    If[z =!= None, tail = ZeroTail[z, xi, kind]];
    dI = -2 Re[Total[resR] + Total[resD]] + tail[[1]];
    dI1 = -2 Re[Total[qR resR] + Total[qD resD]] + tail[[2]];
    <|"xi" -> xi, "F" -> gh["I"] + dI, "alpha" -> (gh["I1"] + dI1)/(gh["I"] + dI),
      "Fthimble" -> gh["I"], "alphaThimble" -> gh["I1"]/gh["I"], "dFStokes" -> dI,
      "dalphaStokes" -> (dI1 - gh["I1"]/gh["I"] dI)/(gh["I"] + dI),
      "resRiemann" -> -2 Re[Total[resR]], "resDyadic" -> -2 Re[Total[resD]], "zeroTail" -> tail[[1]],
      "nRiemann" -> Length[iR], "nDyadic" -> Length[iD], "terminalZero" -> z,
      "q0" -> th["q0"], "S2" -> th["S2"], "drift" -> gh["drift"], "missingNodes" -> gh["missingNodes"],
      "thimble" -> th|>];

(* ::Section:: *)
(*6b. The complete log-periodic part (all wall poles), valid where the right-closed residue series converges*)


(* For r < C_min = Min[abcC] (xi < Log[C_min] ~ -2.92) D(r) is the convergent sum of right residues:
   real powers r^0, r^2, r^(5/2), r^4, ... plus the log-periodic wall terms r^(7 + I gamma_n), r^(15/2 + 2 Pi I k/Log[2]).
   For r > C_max = Max[abcC] (xi > -2.76) it is a convergent series in real powers of 1/r: NO oscillation at all.
   WallOscillation gives the wall part of F and its contribution to alpha. *)
wallQ := wallQ = Join[7 + I gammaList, 15/2 + I dyadList];
wallC[kind_String] := wallC[kind] = Join[RiemannResidue[#, 0., kind] & /@ (7 + I gammaList), DyadicResidue[#, 0., kind] & /@ (15/2 + I dyadList)];
WallOscillation[xi_?NumericQ, kind_String, F_?NumericQ, alpha_?NumericQ] := Module[{e = Exp[xi wallQ] wallC[kind], w, w1},
    w = -2 Re[Total[e]]; w1 = -2 Re[Total[wallQ e]];
    <|"dF" -> w, "dalpha" -> (w1 - alpha w)/F|>];

(* independent check: direct integral along the vertical line Re q = c *)
DirectLine[xi_?NumericQ, kind_String] := Module[{c = If[kind === "f", -0.5, 1.], y},
    1/Pi NIntegrate[Re[Exp[xi (c + I y)] pref[kind] RAn[c + I y] DF0[-1 - c - I y]], {y, 0, 2, 5, 10, 20, 40, 70},
      MaxRecursion -> 30, PrecisionGoal -> 11, AccuracyGoal -> 20]];
DirectLineAlpha[xi_?NumericQ, kind_String] := Module[{c = If[kind === "f", -0.5, 1.], y},
    (1/Pi NIntegrate[Re[(c + I y) Exp[xi (c + I y)] pref[kind] RAn[c + I y] DF0[-1 - c - I y]], {y, 0, 2, 5, 10, 20, 40, 70},
       MaxRecursion -> 30, PrecisionGoal -> 11, AccuracyGoal -> 20])/DirectLine[xi, kind]];


(* ::Section:: *)
(*7. Plot helpers*)


(* poles (red) and zeros (green) of ZR in the q plane *)
PolePtsQ = Join[
    Table[{7., s gammaList[[n]]}, {n, nRiemann}, {s, {1, -1}}] // Flatten[#, 1] &,
    Table[{7.5, s dyadList[[k]]}, {k, nDyadic}, {s, {1, -1}}] // Flatten[#, 1] &,
    {{-1., 0.}, {2.5, 0.}, {5.5, 0.}, {7.5, 0.}},
    Table[{2. n, 0.}, {n, -20, 20}],
    Table[{7.5 + 2 n, 0.}, {n, 1, 10}]];
ZeroPtsQ = Join[
    Table[{6., s gammaList[[n]]}, {n, nRiemann}, {s, {1, -1}}] // Flatten[#, 1] &,
    Table[{6.5 + 2 n, 0.}, {n, 0, 10}]];

FullThimbleQ[th_Association, nPts_: 800] := Module[{pts},
    pts = th["p"] /@ Subdivide[th["eps"], ThimbleEnd[th], nPts];
    Join[Reverse[Conjugate /@ pts], {th["q0"]}, pts]];

PlotThimbles[res_List, {xmin_, xmax_}, {ymin_, ymax_}] := Module[{cols, curves},
    cols = ColorData["Rainbow"] /@ Subdivide[0, 1, Max[Length[res] - 1, 1]];
    curves = Table[{cols[[i]], Thick, Line[ReIm[FullThimbleQ[res[[i]]["thimble"]]]]}, {i, Length[res]}];
    Legended[
     Graphics[{curves,
       Red, PointSize[0.006], Point[Select[PolePtsQ, xmin <= #[[1]] <= xmax && ymin <= #[[2]] <= ymax &]],
       Darker[Green], PointSize[0.005], Point[Select[ZeroPtsQ, xmin <= #[[1]] <= xmax && ymin <= #[[2]] <= ymax &]],
       Table[{cols[[i]], PointSize[0.012], Point[{res[[i]]["q0"], 0}]}, {i, Length[res]}],
       {Dashed, Red, Line[{{7, ymin}, {7, ymax}}]}, {Dashed, Purple, Line[{{7.5, ymin}, {7.5, ymax}}]},
       {Dashed, Darker[Green], Line[{{6, ymin}, {6, ymax}}]}},
      PlotRange -> {{xmin, xmax}, {ymin, ymax}}, Frame -> True, FrameLabel -> {"Re q", "Im q"},
      AspectRatio -> 1, ImageSize -> 700, Background -> White],
     LineLegend[cols, Row[{"log r = ", NumberForm[#["xi"], {3, 2}], ", trapped R/D = ", #["nRiemann"], "/", #["nDyadic"],
          If[#["terminalZero"] =!= None, Row[{", ends on 6+i\[Gamma]", #["terminalZero"]}], ""]}] & /@ res]]];
