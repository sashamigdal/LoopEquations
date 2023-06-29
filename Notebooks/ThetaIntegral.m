(* ::Package:: *)

(* ::Input::Initialization:: *)
ClearAll[H,W, F];


(* ::Input:: *)
(*(* The basic function *)*)


(* ::Input::Initialization:: *)
F[RR_, x_, y_]:=
	1/Sqrt[Det[x -RR - y Im[RR]]];


(* ::Input:: *)
(*(* Ihe inner integral, coming froom the theta function, formula A155 from the Appendix of *)*)


(* ::Input::Initialization:: *)
H[RR_, x_]:=
Block[{S,Q, y,II},
S =F[RR,x,y];
Q = (S -(S/.y->-y))/y;
(S/.y->0)I/2 +1/(2Pi) NIntegrate[ Q,{y,0,Infinity},Exclusions->y==0,
PrecisionGoal->12,WorkingPrecision->16,AccuracyGoal->12, Method->"GaussKronrodRule", MaxRecursion->30]
]//Quiet;


(* ::Input:: *)
(*(* testing on random numbers*)*)


(* ::Input:: *)
(*R1 = ArrayReshape[Table[Random[Complex],{16}],{4,4}];*)
(*R1 = R1 + Transpose[R1];*)


(* ::Input:: *)
(*H[R1,1.]*)


(* ::Input:: *)
(*(*The restricted group integral, formula A 148 fropm the Appendix*)*)


(* ::Input::Initialization:: *)
W[RR_]:=
Block[{ x},
NIntegrate[Exp[I x]/(2 Pi) H[RR,x],{x,-Infinity,Infinity},
PrecisionGoal->16,WorkingPrecision->20,AccuracyGoal->12, Method->"DoubleExponentialOscillatory", MaxRecursion->30]
]//Quiet;


(* ::Input:: *)
(**)
(*(* testing on random number *)*)
(*W[ R1]*)


(* ::Input:: *)
(**)
(*(* testing on random numbers which proiduces |W| > 1 with an old buggy version *)*)
