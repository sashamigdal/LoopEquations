(* ::Package:: *)

(* ::Input::Initialization:: *)
W[MM_]/;Dimensions[MM]=={4,4}:=
Block[{Q,RR,TT,tt,W,V,QI,x,y,z},TT=Im[MM];
RR=Re[MM];
{tt,W}=Eigensystem[TT];
RR=W . RR . Transpose[W];
V={x,y,z};
Q=Append[2 V,(1-V . V)]/(1+V . V);
QI=Simplify[(Q^2) . tt];
4/Pi^2 NIntegrate[Exp[I Q . RR . Q-QI] Boole[Numerator[QI]>0]/(1+V . V)^3,
{x,-Infinity,Infinity},{y,-Infinity,Infinity},{z,-Infinity,Infinity}]
]//Quiet;


(* ::Input:: *)
(*R1 = ArrayReshape[Table[Random[Complex],{16}],{4,4}];*)


(* ::Input:: *)
(*R1 = R1 + Transpose[R1];*)


(* ::Input:: *)
(*Max[Abs[Im[R1]]]*)


(* ::Input:: *)
(* W[R1]*)


(* ::Input::Initialization:: *)
 WW[{M1_, M2_, {t0_, t1_, steps_}}]/;Dimensions[M1]=={4,4}&& Dimensions[M2]=={4,4}:=
ParallelTable[(W[M1/Sqrt[t]] + W[M2/Sqrt[t]])/2,{t,t0, t1, (t1-t0)/steps}]


(* ::Input:: *)
(**)
(*R2 = ArrayReshape[Table[Random[Complex],{16}],{4,4}];*)
(*R2 = R2+ Transpose[R2];*)


(* ::Input:: *)
(*WW[{R1, R2, {1, 2, 10}}]*)


(* ::Input:: *)
(*ClearSystemCache[];Timing[WW[{R1,R2,{1,2,10}}]]*)
