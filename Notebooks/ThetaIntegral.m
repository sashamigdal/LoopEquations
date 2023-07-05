(* ::Package:: *)

(* ::Input::Initialization:: *)
ClearAll[W0];


(* ::Input:: *)
(*(*The unrestricted Fourier  integral*)*)


(* ::Input:: *)
(**)


(* ::Input::Initialization:: *)
W0[R_, factor_]:=
Block[{shift,roots,x, P,ims, L, Gc},
P = Det[ x IdentityMatrix[4]- R ];
(*Print[P];*)
roots =x/.NSolve[ P==0,x];
(*Print[roots];*)
shift = Min[0,factor Min[Im[roots]]];
(* 
we are shifting the integration contour below the lowest singularity.
This is the analytic continuation of the above integral .
We take care of correct branch of the square root, making the phase the same as at R=0.
The result does not depend upon the shift factor, as the complex analysis tells it should not
*)
(*Print[shift];*)
Gc=Compile[{{y,_Complex}},-Exp[I y]/(y^2Sqrt[ Det[ IdentityMatrix[4]- R /y]])/(2 Pi)];
(*Return[G];*)
NIntegrate[Gc[x+ I shift],{x,-Infinity,Infinity},
PrecisionGoal->10,WorkingPrecision->16,AccuracyGoal->10, Method->"GaussKronrodRule", 
MaxRecursion->30]/(2 Pi)
]//Quiet;


(* ::Input:: *)
(*           *)


(* ::Input:: *)
(*R1 = ArrayReshape[Table[Random[Complex],16],{4,4}];*)
(*R1 += Transpose[R1];*)


(* ::Input:: *)
(*Table[W0[R1,f],{f,3,5,0.5}]//MatrixForm*)


(* ::Input:: *)
(*(**)
(*The restricted Fourier  integral as analytic continuation.*)
(*The nomerical precision is very low, leading to spurious results,     strongly dependent upon the shift of the path below the lowest singulatity.*)
(*According to complex analysis, there should be no such dependence. *)
(*In case of unrestricted fourier integral, there is a very wek dependence, *)
(*   in higher digits.*)
(* .*)*)


(* ::Input::Initialization:: *)
LowestSingularity[R_]:=
Min[0.,
Block[{P,x,y, roots},
P = Det[ x IdentityMatrix[4]- R - y Im[R] ];
roots =x/.NSolve[ P==0,x];
NMinimize[Min[Im[roots]],y][[1]]
]
];



(* ::Input:: *)
(*LowestSingularity[R1]*)


(* ::Input::Initialization:: *)
WR[R_,  factor_]:=
Block[{Gc, shift},
shift = factor LowestSingularity[R];
Gc=Compile[{{x,_Complex},{y, _Complex}},-Exp[I x]/(x^2Sqrt[ Det[ IdentityMatrix[4]- (R  +y Im[R])/x]])/y/( (2Pi)(2 Pi I))];NIntegrate[Gc[x+ I shift,y],{y,-Infinity,Infinity}, {x, -Infinity, Infinity},
PrecisionGoal->10,WorkingPrecision->16,AccuracyGoal->10, Method->"GaussKronrodRule", MaxRecursion->30]
]


(* ::Input:: *)
(*WR[R1,2.5]*)


(* ::Input:: *)
(*WR[2*R1,3.5]*)


(* ::Input:: *)
(*WR[10*R1,1.5]*)


(* ::Input:: *)
(* Get[StringJoin[NotebookDirectory[],"SaddlePointFourierIntegral.m"]]*)


(* ::Input:: *)
(*Max[Abs[R1]]*)


(* ::Input:: *)
(*WR[R_]:=If[Max[Abs[R]] >1, W[R], WR[R,1.5]]*)


(* ::Input:: *)
(*WR[R1/Max[Abs[R1]]]*)


(* ::Input:: *)
(*W[R1/Max[Abs[R1]]]*)
