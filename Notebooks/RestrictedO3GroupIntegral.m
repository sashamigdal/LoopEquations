(* ::Package:: *)

(* ::Input::Initialization:: *)
Get["Notebooks/PadeBorel.m"];


(* ::Input:: *)
(*(*Goal[\[Epsilon]_] := Integrate[Exp[- F[{x,y}]/\[Tau]^2], {x - Infinity,Infinity},{y,-Infinity,Infinity}];at \[Tau] -> 0*)*)


(* ::Input:: *)
(*PadeBorel[t , 1+Sum[x^n n!,{n,1,12}],x , 5]*)


(* ::Input:: *)
(*PadeBorel[1. , 1+Sum[x^n n!,{n,1,12}],x , 5]*)


(* ::Input::Initialization:: *)
ClearAll[x,y,u,v,\[Tau], \[Epsilon]];


(* ::Input::Initialization:: *)
Dif[s1_,s2_, tol_]:= N[Norm[({x,y}/.s1) - ({x,y}/.s2) ]] < tol;


(* ::Input::Initialization:: *)
Dupl[rules_, tol_] :=Transpose[Position[Table[Dif[rules[[k]], rules[[k-1]],tol],{k,2,Length[rules]}],True]][[1]]


(* ::Input::Initialization:: *)
RemElems2[A_, pos_] := A[[#]]& /@ DeleteElements[Range[Length[A]],pos];


(* ::Input::Initialization:: *)
RemoveRepeated[rules_, tol_] := RemElems2[rules,Dupl[rules, tol] ] ;


(* ::Input::Initialization:: *)
SaddlePointExpansion[F_, L_]:=
Block[{saddlePoint,ref,best,sol, \[Tau], r0,x,y,u,v,f0, Q, F3, singular, expansion,
GenFun,SymRule, WickRule, res, E2},
saddlePoint = RemoveRepeated[NSolve[Grad[F[{x,y}],{x,y}]== {0,0},{x,y}],10^(-9)];
If[Length[saddlePoint] ==0, Return[0]];
saddlePoint = Select[saddlePoint, Re[F[{x,y}/.# ]] >=0&];
ref =Re[F[{x,y}]]/.saddlePoint;
best = Position[ref,Min[ref]][[1,1]];
sol = saddlePoint[[best]];
Print["sol=", sol, ", f0=", F[{x,y}]/.sol];
(*Return[sol];*)
Assuming[{Im[u \[Tau]] ==0, Im[v \[Tau]]==0},
(*Print["sol=",sol];*)
r0 = {x,y}/.sol;
Print["r0=",r0];
f0 = Simplify[F[r0]];
Print["f0=",f0];
Q=Grad[ Grad[F[{x,y}],{x,y}],{x,y}]/.sol;
Print["Hessian=",MatrixForm[Q]];
singular= 2 Pi \[Tau]^2Exp[-f0/\[Tau]^2] Det[Q]^(-1/2);
(*Print["singular= ", singular];*)
F3 =Collect[Normal[Series[(F[{x+ u \[Tau],y + v \[Tau]}]-F[{x,y}])/.sol,{\[Tau],0,L+2}]],\[Tau]];
(*Print["F3=",F3];*)
F3 =Normal[Collect[F3[[3;;]]/\[Tau]^2,\[Tau]] + O[\[Tau]]^L];
(*Print["F3=",F3];*)
SymRule = A_ \[Tau]^n_. :>0 /;OddQ[n];
expansion =Collect[Normal[ Exp[-F3] + O[\[Tau]]^L],\[Tau]]/.SymRule;
(*Print["expansion=",expansion];*)
GenFun = Exp[1/2Expand[{u,v} . Inverse[Q] . {u,v}]];
(*Print["GenFun=",GenFun];*)
WickRule = {
u^n_. v^m_. :> D[GenFun,{u,n},{v,m}]
};
E2 = (((ExpandAll[expansion]/.WickRule)/.{u->0,v->0}))/.\[Tau]->x;
(*Print["E2=",E2];*)
(singular/.\[Tau]->1) PadeBorel[1.,E2,x,L/2]
]
];


(* ::Input:: *)
(*\!\(TraditionalForm\`\**)
(*OverscriptBox[*)
(*StyleBox["W", "TI"], "~"] \((\**)
(*OverscriptBox[*)
(*StyleBox["R", "TI"], "^"])\) == *)
(*\*FractionBox[\(1\), *)
(*SuperscriptBox[\(\[Pi]\), \(2\)]] *)
(*\*SubsuperscriptBox[\(\[Integral]\), \(+-\[ScriptDotlessI]\[Infinity]\), \(++\[ScriptDotlessI]\[Infinity]\)]\**)
(*FractionBox[*)
(*RowBox[{*)
(*StyleBox["d", "TI"], "\[Tau]", "exp", "\[Tau]"}], *)
(*RowBox[{"2", "\[Pi]\[ScriptDotlessI]"}]] *)
(*\*SubsuperscriptBox[\(\[Integral]\), \(+-\[ScriptDotlessI]\[Infinity]\), \(++\[ScriptDotlessI]\[Infinity]\)]\**)
(*FractionBox[*)
(*RowBox[{*)
(*StyleBox["d", "TI"], "\[Lambda]"}], *)
(*RowBox[{"2", "\[Pi]\[ScriptDotlessI]\[Lambda]"}]]\n\[Integral]\**)
(*SuperscriptBox[*)
(*StyleBox["d", "TI"], "4"] \**)
(*StyleBox["q", "TI"] exp \(-\**)
(*SubscriptBox[*)
(*StyleBox["q", "TI"], "\[Alpha]"] \**)
(*SubscriptBox[*)
(*StyleBox["q", "TI"], "\[Beta]"] \(( *)
(*\*SubscriptBox[\(\[Tau]\[Delta]\), \(\[Alpha]\[Beta]\)] - \[Lambda]  \[GothicCapitalI] \**)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], "\[Alpha]\[Beta]"] - \[ScriptDotlessI] \**)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], "\[Alpha]\[Beta]"])\)\) == \n*)
(*\*SubsuperscriptBox[\(\[Integral]\), \(+-\[ScriptDotlessI]\[Infinity]\), \(++\[ScriptDotlessI]\[Infinity]\)]\**)
(*FractionBox[*)
(*RowBox[{*)
(*StyleBox["d", "TI"], "\[Tau]", "exp", "\[Tau]"}], *)
(*RowBox[{"2", "\[Pi]\[ScriptDotlessI]"}]] *)
(*\*SubsuperscriptBox[\(\[Integral]\), \(+-\[ScriptDotlessI]\[Infinity]\), \(++\[ScriptDotlessI]\[Infinity]\)]\**)
(*FractionBox[*)
(*RowBox[{*)
(*StyleBox["d", "TI"], "\[Lambda]"}], *)
(*RowBox[{"2", "\[Pi]\[ScriptDotlessI]\[Lambda]"}]] \**)
(*SuperscriptBox[*)
(*RowBox[{"(", *)
(*RowBox[{*)
(*UnderoverscriptBox["\[Product]", *)
(*RowBox[{*)
(*StyleBox["k", "TI"], "==", "1"}], "4",*)
(*LimitsPositioning->True], "(", *)
(*RowBox[{"\[Tau]", "-", "\[Lambda]", "\[GothicCapitalI]", *)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], *)
(*StyleBox["k", "TI"]], "-", "\[ScriptDotlessI]", *)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], *)
(*StyleBox["k", "TI"]]}], ")"}], ")"}], *)
(*RowBox[{"-", *)
(*FractionBox["1", "2"]}]]\)*)


(* ::Input:: *)
(*F0[{\[Tau]_,\[Lambda]_}] :=*)
(*With[ {RR = {-0.1 + 2 I, -0.2 -0.01 I, -0.3 + 0.05 I, -0.1 - 0.07 I}},*)
(* 2 Log[2 Pi I]-\[Tau] -Log[\[Lambda]] +1/2 Sum[Log[\[Tau] -\[Lambda] Im[RR[[i]]] - RR[[i]]],{i,4}]]*)


(* ::Input:: *)
(*F0[{x,y}]*)


(* ::Input:: *)
(**)


(* ::Input:: *)
(*sp=SaddlePointExpansion[F0, 18]*)


(* ::Input::Initialization:: *)
W[R_]:=
Block[{F0},
F0[{\[Tau]_,\[Lambda]_}] :=2 Log[2 Pi I]-\[Tau] -Log[\[Lambda]] +1/2 Sum[Log[\[Tau] -\[Lambda] Im[R[[i]]] - R[[i]]],{i,4}];
SaddlePointExpansion[F0, 18]
]
