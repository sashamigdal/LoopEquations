(* ::Package:: *)

(* ::Input::Initialization:: *)
(*(**)
(*Solution of linearized equations near the existing solution*)
(*for three vertices, with F0 and F3 fixed, and F1, F2 movind *)
(*along the solution of these equations.*)
(*Purpose: to find the null space of thgese linearized equations k = 0,1,2*)
(*(Subscript[Overscript[F, \[RightVector]], k+1]-Subscript[Overscript[F, \[RightVector]], k])^2\[LongEqual]1;*)
(*(\!\(\**)
(*SubsuperscriptBox[*)
(*OverscriptBox[*)
(*StyleBox["F", "TI"], "\[RightVector]"], *)
(*RowBox[{*)
(*StyleBox["k", "TI"], "+", "1"}], "2"] - \**)
(*SubsuperscriptBox[*)
(*OverscriptBox[*)
(*StyleBox["F", "TI"], "\[RightVector]"], *)
(*StyleBox["k", "TI"], "2"] - \[ScriptDotlessI]\))^2\[LongEqual](Subscript[Overscript[F, \[RightVector]], k+1]+Subscript[Overscript[F, \[RightVector]], k])^2-1*)
(**)*)


(* ::Input::Initialization:: *)
Q= {q0,q1,q2}


(* ::Input::Initialization:: *)
F= {F0, F0+q0,F3-q2,F3}


(* ::Input::Initialization:: *)
(* symbolic opreations on vectors *)


(* ::Input::Initialization:: *)
VDV[V_] := Expand[V^2]/.{A_Symbol * B_Symbol-> A . B, C_Symbol^2 -> C . C}


(* ::Input::Initialization:: *)
VDW[V_,W_] := Expand[V W]/.{A_Symbol * B_Symbol-> A . B, C_Symbol^2 -> C . C, A_Symbol * Dot[B_,C_] :> A . B . C}


(* ::Input::Initialization:: *)
VDV[3 x +5y]


(* ::Input::Initialization:: *)
VDW[3 x + 2 y, y + z]


(* ::Input::Initialization:: *)

(* three infinitesimal elements of  Lie algebra for O(3)*)


(* ::Input::Initialization:: *)
dt = {dt0,dt1,dt2}


(* ::Input::Initialization:: *)
dQ =Table[\[Lambda] dt[[i]] . Q[[i]],{i,3}]


(* ::Input::Initialization:: *)
(* The variations of all four vertices*)


(* ::Input::Initialization:: *)
dF = {0,dQ[[1]],dQ[[1]] + dQ[[2]],0}


(* ::Input::Initialization:: *)
(* Shifted vertices*)


(* ::Input::Initialization:: *)
F += dF


(* ::Input::Initialization:: *)
(* some substitutions needed to simplify equations*)


(* ::Input::Initialization:: *)
Sub1 = {A_Symbol * B_Symbol-> A . B, C_Symbol^2 -> C . C};


(* ::Input::Initialization:: *)
Sub2 = {A_Symbol B_Symbol . C_Symbol :> A . B . C};


(* ::Input::Initialization:: *)
Sub3 = {Dot[0,x_]:>0, Dot[X_,0] :> 0, Dot[X_,1]:> X, Dot[1,X_] :> X};


(* ::Input::Initialization:: *)
Sub4 =
{
q0 . dt0 . q0 ->0, q0 . q0 ->1,
q1 . dt1 . q1->0, q1 . q1->1,
q2 . dt2 . q2 ->0, q2 . q2 ->1
};


(* ::Input::Initialization:: *)
AllSubs = Join[Sub1,Sub2,Sub3,Sub4];


(* ::Input::Initialization:: *)
Eqs = Normal[Table[ (VDV[F[[k+1]]]- VDV[F[[k]]]- I)^2-VDV[F[[k]] + F[[k+1]]]+1,{k,3}] + O[\[Lambda]]^2]//.AllSubs



(* ::Input::Initialization:: *)
EqV0 = dQ[[1]]+ dQ[[2]]+dQ[[3]];


(* ::Input::Initialization:: *)
TMP= {A0,A1,A2}


(* ::Input::Initialization:: *)
Eq0 = \[Lambda] Table[VDW[TMP[[i]],EqV0 /.\[Lambda]->1],{i,3}]/.{A0->Ort[0],A1->Ort[1],A2->Ort[2]}


(* ::Input::Initialization:: *)
Eqs = Join[Eqs,Eq0];


(* ::Input::Initialization:: *)
Dimensions[Eqs]


(* ::Input::Initialization:: *)
GradEq=D[Eqs,\[Lambda]]


(* ::Input::Initialization:: *)
Sub5[X_, Y_] :={Dot[A_, X,B_] :> Y[l] A . E3[l] . B, Dot[ X,B_] :> Y[l] E3[l] . B}


(* ::Input::Initialization:: *)
GradEqPlus={D[GradEq/.Sub5[dt0,t0],t0[l]],D[GradEq/.Sub5[dt1,t1],t1[l]],D[GradEq/.Sub5[dt2,t2],t2[l]]}
