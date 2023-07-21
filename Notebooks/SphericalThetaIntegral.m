(* ::Package:: *)

(* ::Input:: *)
(*ClearAll[W];*)


(* ::Input::Initialization:: *)
PosCond1[{a_,b_}]:=Piecewise[{{Max[0,-a/b],b>0}},0];
PosCond2[{a_,b_}]:=Piecewise[{{1,b>0}},Min[1,-a/b]];


(* ::Input::Initialization:: *)
W[MM_]/;Dimensions[MM]=={4,4}:=Block[{Q,RR,TT,tt,W,QTTQ,u,v,w,Jac,Vol},
TT=Im[MM];
{tt,W}=Eigensystem[TT];
RR=Transpose[W] . Re[MM] . W;
Q={Sin[w] Cos[u],Sin[w] Sin[u],Cos[w] Cos[v],Cos[w] Sin[v]};
QTTQ=Evaluate[(Q^2) . tt];
Jac=Sin[w] Cos[w];
Vol=Integrate[Jac,{w,0,Pi/2}] (2 Pi)^2;
NIntegrate[Exp[I Q . RR . Q-QTTQ] Boole[QTTQ>0] Jac/Vol,{w,0,Pi/2},{u,0,2 Pi},{v,0,2 Pi},
Method->"LocalAdaptive", AccuracyGoal->8]
];


(* ::Input::Initialization:: *)
Winv[MM_]/;Dimensions[MM]=={4,4}:=Block[{Q,RR,TT,tt,W,QTTQ,u,v,w,Jac,Vol},
Q={Sin[w] Cos[u],Sin[w] Sin[u],Cos[w] Cos[v],Cos[w] Sin[v]};
QTTQ=Evaluate[Q . Im[MM] . Q];
Jac=Sin[w] Cos[w];
Vol=Integrate[Jac,{w,0,Pi/2}] (2 Pi)^2;
NIntegrate[Exp[I Q . MM . Q] Boole[QTTQ>0] Jac/Vol,{w,0,Pi/2},{u,0,2 Pi},{v,0,2 Pi},
Method->"LocalAdaptive", AccuracyGoal->8]
];


(* ::Input:: *)
(*R1 = ArrayReshape[Table[Random[Complex],{16}],{4,4}];*)


(* ::Input:: *)
(*R1 = R1 + Transpose[R1];*)


(* ::Input:: *)
(*Max[Abs[Im[R1]]]*)


(* ::Input:: *)
(* W[R1]//Timing*)


(* ::Input:: *)
(*{23.602027`,0.09799485123761706` +0.15761670118563176` I}*)


(* ::Input:: *)
(*{5.591456`,0.09801320976651084` +0.15758849096271882` I}*)


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


(* ::Input:: *)
(*Winv[{{(-1.0206513619464603)+(-0.6799689154646605) I,(2.787881129843189)+(0.6403894627859661) I,(1.9524760902937979)+(-0.188782075204682) I,(-1.4244736627068686)+(-0.9128020711169895) I},{(2.787881129843189)+(0.6403894627859661) I,(-1.730225288380606)+(1.529355107811821) I,(-2.3159935023421836)+(0.37878899346095535) I,(-0.6625971715098803)+(0.07859164232209265) I},{(1.9524760902937979)+(-0.188782075204682) I,(-2.3159935023421836)+(0.37878899346095535) I,(-3.72878017326537)+(-0.18313207758137823) I,(0.21432244577252924)+(-1.7381508714002714) I},{(-1.4244736627068686)+(-0.9128020711169895) I,(-0.6625971715098803)+(0.07859164232209265) I,(0.21432244577252924)+(-1.7381508714002714) I,(-2.7455348836685913)+(-1.943620600734952) I}}*)
(*]*)
