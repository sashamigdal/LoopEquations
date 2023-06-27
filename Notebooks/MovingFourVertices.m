(* ::Package:: *)

(* ::Input::Initialization:: *)
(*(**)
(*Solution of linearized equations near the existing solution*)
(*for four vertices, with F0 and F4 fixed, and F1, F2, F3 movind *)
(*along the solution of these equations.*)
(*Purpose: to find the null space of thgese linearized equations k = 0,1,2,3*)
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
(* symbolic opreations on vectors *)


(* ::Input::Initialization:: *)
ClearAll[VDV, VDW];


(* ::Input::Initialization:: *)
SubDF ={dF[k_]* dF[l_]:>0,dF[k_]^2:>0,F[k_]*F[l_]:> F[k] . F[l],F[k_]*dF[l_]:> F[k] . dF[l], dF[k_]*F[l_]:> F[l] . dF[k],F[k_]^2:>  F[k] . F[k]}


(* ::Input::Initialization:: *)
VDV[V_] := Expand[V*V]/.SubDF


(* ::Input::Initialization:: *)
VDW[V_,W_] := Expand[V*W]/.SubDF


(* ::Input::Initialization:: *)

(* three infinitesimal elements of  Lie algebra for O(3)*)


(* ::Input::Initialization:: *)
(* The variations of all four vertices*)


(* ::Input::Initialization:: *)
(* Shifted vertices*)


(* ::Input::Initialization:: *)
FF = {F[0], (F[#] + \[Lambda] dF[#] )&/@ Range[3],F[4]}//Flatten


(* ::Input::Initialization:: *)
Eqs = Normal[Flatten[Table[ {VDV[FF[[k]] - FF[[k+1]]]-1,Expand[(VDV[FF[[k+1]]]- VDV[FF[[k]]]- I)^2-VDV[FF[[k]] + FF[[k+1]]]+1]},{k,4}] ]+ O[\[Lambda]]^2]//.SubDF



(* ::Input::Initialization:: *)
Dimensions[Eqs]


(* ::Input::Initialization:: *)
Lineq =D[Eqs,\[Lambda]]//FullSimplify


(* ::Input:: *)
(*MatrixForm[Lineq]*)
(**)
