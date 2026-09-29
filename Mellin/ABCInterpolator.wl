(* ::Package:: *)

ClearAll["Global`*"]


(* ::Input::Initialization:: *)
gg[r_,\[CapitalDelta]_] :=(-2+2 Cos[Sqrt[r (-1+\[CapitalDelta])]] Cos[Sqrt[r \[CapitalDelta]]]+((-1+2 \[CapitalDelta]) Sin[Sqrt[r (-1+\[CapitalDelta])]] Sin[Sqrt[r \[CapitalDelta]]])/Sqrt[(-1+\[CapitalDelta]) \[CapitalDelta]])/r^2;

action[ r_,\[CapitalDelta]_]:= (Sqrt[r] (-8 Sqrt[r] Sqrt[-1+\[CapitalDelta]] Sqrt[\[CapitalDelta]] Cos[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]-Sqrt[\[CapitalDelta]])]+8 Sqrt[r] Sqrt[-1+\[CapitalDelta]] Sqrt[\[CapitalDelta]] Cos[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]+Sqrt[\[CapitalDelta]])]+4 Sqrt[r] (-1+2 \[CapitalDelta]-\[CapitalDelta] Cos[2 Sqrt[r] Sqrt[-1+\[CapitalDelta]]]-(-1+\[CapitalDelta]) Cos[2 Sqrt[r] Sqrt[\[CapitalDelta]]])-8 Sqrt[-1+\[CapitalDelta]] \[CapitalDelta] Sin[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]-Sqrt[\[CapitalDelta]])]-Sqrt[-1+\[CapitalDelta]] Sin[2 Sqrt[r] (Sqrt[-1+\[CapitalDelta]]-Sqrt[\[CapitalDelta]])]+4 Sqrt[-1+\[CapitalDelta]] \[CapitalDelta] Sin[2 Sqrt[r] (Sqrt[-1+\[CapitalDelta]]-Sqrt[\[CapitalDelta]])]-8 Sqrt[-1+\[CapitalDelta]] \[CapitalDelta] Sin[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]+Sqrt[\[CapitalDelta]])]-Sqrt[-1+\[CapitalDelta]] Sin[2 Sqrt[r] (Sqrt[-1+\[CapitalDelta]]+Sqrt[\[CapitalDelta]])]+4 Sqrt[-1+\[CapitalDelta]] \[CapitalDelta] Sin[2 Sqrt[r] (Sqrt[-1+\[CapitalDelta]]+Sqrt[\[CapitalDelta]])]+2 Sqrt[-1+\[CapitalDelta]] Sin[2 Sqrt[r] Sqrt[-1+\[CapitalDelta]]]+Sqrt[\[CapitalDelta]] (-8 (-1+\[CapitalDelta]) Sin[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]-Sqrt[\[CapitalDelta]])]+(-3+4 \[CapitalDelta]) Sin[2 Sqrt[r] (Sqrt[-1+\[CapitalDelta]]-Sqrt[\[CapitalDelta]])]+8 (-1+\[CapitalDelta]) Sin[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]+Sqrt[\[CapitalDelta]])]+(3-4 \[CapitalDelta]) Sin[2 Sqrt[r] (Sqrt[-1+\[CapitalDelta]]+Sqrt[\[CapitalDelta]])]+2 Sin[2 Sqrt[r] Sqrt[\[CapitalDelta]]])))/(16 (-1+\[CapitalDelta]) \[CapitalDelta] (Cos[Sqrt[r] Sqrt[-1+\[CapitalDelta]]]-Cos[Sqrt[r] Sqrt[\[CapitalDelta]]])^2);

r0data = Block[{data={}, delta = 0.001, r0 = -16.7},
d = delta;
While[d < 0.98,
r0 =Re[ r/.FindRoot[gg[r,d],{r, r0}][[1]]]//Quiet;
AppendTo[data,{d,r0}];
d += delta;
];
data
];
r2data = Block[{data={}, delta = 0.001, r0 =44.74657089612265`},
d = 0.5;
While[d > 0.05,
r0 =Re[ r/.FindRoot[gg[r,d],{r, r0}][[1]]]//Quiet;
AppendTo[data,{d,r0}];
d -= delta;
];
data
];
ClearAll[r,d];
interp0 = Interpolation[r0data];
interp2 = Interpolation[r2data];
(*\[CapitalDelta]1=Re[d/.FindRoot[action[interp2[d],d]- action[interp0[d],d],{d,0.156}][[1]]];
\[CapitalDelta]2=Re[d/.FindRoot[action[interp2[d],d]- action[interp0[d],d],{d,0.42}][[1]]];*)
\[CapitalDelta]1=0.15714261196307505` ;
\[CapitalDelta]2=0.4301495990065511` ;
cons[\[CapitalDelta]_, r_] :=-((2 \[CapitalDelta] Cos[2 Sqrt[r] Sqrt[-1+\[CapitalDelta]]]+1/Sqrt[r] (2 Sqrt[r] (1-2 \[CapitalDelta])-4 Sqrt[-1+\[CapitalDelta]] \[CapitalDelta] Sin[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]-Sqrt[\[CapitalDelta]])]-4 Sqrt[-1+\[CapitalDelta]] \[CapitalDelta] Sin[Sqrt[r] (Sqrt[-1+\[CapitalDelta]]+Sqrt[\[CapitalDelta]])]+Sqrt[-1+\[CapitalDelta]] Sin[2 Sqrt[r] Sqrt[-1+\[CapitalDelta]]]+Cos[2 Sqrt[r] Sqrt[\[CapitalDelta]]] (2 Sqrt[r] (-1+\[CapitalDelta])+Sqrt[-1+\[CapitalDelta]] (-1+4 \[CapitalDelta]) Sin[2 Sqrt[r] Sqrt[-1+\[CapitalDelta]]])+8 Sqrt[r] Sqrt[-1+\[CapitalDelta]] Sqrt[\[CapitalDelta]] Sin[Sqrt[r] Sqrt[-1+\[CapitalDelta]]] Sin[Sqrt[r] Sqrt[\[CapitalDelta]]]+Sqrt[\[CapitalDelta]] (8 (-1+\[CapitalDelta]) Cos[Sqrt[r] Sqrt[-1+\[CapitalDelta]]] Sin[Sqrt[r] Sqrt[\[CapitalDelta]]]+(1+(3-4 \[CapitalDelta]) Cos[2 Sqrt[r] Sqrt[-1+\[CapitalDelta]]]) Sin[2 Sqrt[r] Sqrt[\[CapitalDelta]]])))/(16 (-1+\[CapitalDelta]) \[CapitalDelta] (Cos[Sqrt[r] Sqrt[-1+\[CapitalDelta]]]-Cos[Sqrt[r] Sqrt[\[CapitalDelta]]])^2));
\[Alpha]\[Alpha][\[CapitalDelta]_,r_]:= (r (\[CapitalDelta] Sin[Sqrt[r (-1+\[CapitalDelta])]]-Sqrt[(-1+\[CapitalDelta]) \[CapitalDelta]] Sin[Sqrt[r \[CapitalDelta]]]) (\[CapitalDelta] Cos[Sqrt[r \[CapitalDelta]]] Sin[Sqrt[r (-1+\[CapitalDelta])]]-Sqrt[(-1+\[CapitalDelta]) \[CapitalDelta]] Cos[Sqrt[r (-1+\[CapitalDelta])]] Sin[Sqrt[r \[CapitalDelta]]]))/((-1+\[CapitalDelta]) \[CapitalDelta]^2 (Cos[Sqrt[r (-1+\[CapitalDelta])]]-Cos[Sqrt[r \[CapitalDelta]]])^2);
f[\[Omega]_,  r_, \[CapitalDelta]_] := -r Cos[Sqrt[\[Omega]]/2]+r Cos[1/2 (1-2 \[CapitalDelta]) Sqrt[\[Omega]]]+(-1+\[CapitalDelta]) \[CapitalDelta] Sqrt[\[Omega]] (r+(-1+\[CapitalDelta]) \[CapitalDelta] \[Omega]) Sin[Sqrt[\[Omega]]/2];
IQ[ \[CapitalDelta]_] :=
Module[{r,L,S,J, Q, \[Omega], Phi, FF, eps = 0.1},
r = interp2[\[CapitalDelta]];
(*Print[r];*)
Phi =D[Log[ f[\[Omega],r,\[CapitalDelta]]/(Cos[Sqrt[\[Omega]]/2] \[Omega]^(3/2))],\[Omega]];
(*Print[{Phi/.{\[Omega]->eps-I L},Phi/.{\[Omega]->eps+I L}}];*)
NIntegrate[ Re[(Phi Log[\[Omega]])/.\[Omega]->eps+I x],{x,-Infinity, Infinity},
Method->"DoubleExponentialOscillatory"]
]//Quiet;
(*qtab = Table[{\[CapitalDelta],IQ[ \[CapitalDelta]]},{\[CapitalDelta],\[CapitalDelta]1,\[CapitalDelta]2,0.00005}];
iqinterp = Interpolation[qtab];
DumpSave[NotebookDirectory[]<>"iqinterp.mx",iqinterp];*)
Get[NotebookDirectory[]<>"iqinterp.mx"];
ClearAll[r];
ABC[ \[CapitalDelta]_] :=
Module[{r,L,S,J, Q},
r = interp2[\[CapitalDelta]];
(*Print[r];*)
L = Re[action[r,\[CapitalDelta]]];
(*Print[L];*)
S= Re[\[Alpha]\[Alpha][\[CapitalDelta], r]];
(*Print[S];*)
J = Re[cons[\[CapitalDelta], r]];
(*Print[J];*)
Q = Exp[1/(8 Pi)iqinterp[\[CapitalDelta]]];
Q {2(r-6)/(r+12),J/S,L/( 2 Pi S)}
]//Quiet;
