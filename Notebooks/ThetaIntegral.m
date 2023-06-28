(* ::Package:: *)

(* ::Input:: *)
(**)
(*\!\(TraditionalForm\`\(\**)
(*OverscriptBox[*)
(*StyleBox["W", "TI"], "~"] \((\**)
(*OverscriptBox[*)
(*StyleBox["R", "TI"], "^"])\)\(==\)\(\n\)*)
(*\*SubsuperscriptBox[\(\[Integral]\), \(-\[Infinity]\), \(+\[Infinity]\)]\**)
(*FractionBox[*)
(*RowBox[{*)
(*StyleBox["d", "TI"], *)
(*StyleBox["x", "TI"], "exp", *)
(*RowBox[{"\[ScriptDotlessI]", *)
(*StyleBox["x", "TI"]}]}], *)
(*RowBox[{"2", "\[Pi]"}]] *)
(*\*SubsuperscriptBox[\(\[Integral]\), \(-\[Infinity]\), \(+\[Infinity]\)]\**)
(*FractionBox[*)
(*StyleBox[*)
(*RowBox[{"d", "y"}], "TI"], *)
(*RowBox[{"2", "\[Pi]\[ScriptDotlessI]", *)
(*RowBox[{"(", *)
(*RowBox[{*)
(*StyleBox["y", "TI"], "-", "\[ScriptDotlessI]\[Epsilon]"}], ")"}]}]] \**)
(*SuperscriptBox[*)
(*RowBox[{"(", *)
(*RowBox[{*)
(*UnderoverscriptBox["\[Product]", *)
(*RowBox[{*)
(*StyleBox["k", "TI"], "==", "1"}], "4",*)
(*LimitsPositioning->True], "(", *)
(*RowBox[{*)
(*StyleBox["x", "TI"], "-", *)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], *)
(*StyleBox["k", "TI"]], "-", *)
(*StyleBox["y", "TI"], "\[GothicCapitalI]", *)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], *)
(*StyleBox["k", "TI"]]}], ")"}], ")"}], *)
(*RowBox[{*)
(*RowBox[{"-", "1"}], "/", "2"}]]\(==\)\(\n\)*)
(*\*SubsuperscriptBox[\(\[Integral]\), \(-\[Infinity]\), \(+\[Infinity]\)]\**)
(*FractionBox[*)
(*RowBox[{*)
(*StyleBox["d", "TI"], *)
(*StyleBox["x", "TI"], "exp", *)
(*RowBox[{"\[ScriptDotlessI]", *)
(*StyleBox["x", "TI"]}]}], *)
(*RowBox[{"2", "\[Pi]"}]] \**)
(*SuperscriptBox[*)
(*RowBox[{"(", *)
(*RowBox[{*)
(*UnderoverscriptBox["\[Product]", *)
(*RowBox[{*)
(*StyleBox["k", "TI"], "==", "1"}], "4",*)
(*LimitsPositioning->True], "(", *)
(*RowBox[{*)
(*StyleBox["x", "TI"], "-", *)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], *)
(*StyleBox["k", "TI"]]}], ")"}], ")"}], *)
(*RowBox[{*)
(*RowBox[{"-", "1"}], "/", "2"}]] *)
(*StyleBox["H", "TI"] \((\**)
(*FractionBox[*)
(*RowBox[{"\[GothicCapitalI]", *)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], *)
(*StyleBox["k", "TI"]]}], *)
(*RowBox[{*)
(*StyleBox["x", "TI"], "-", *)
(*SubscriptBox[*)
(*StyleBox["R", "TI"], *)
(*StyleBox["k", "TI"]]}]])\)\(;\)\(\n\)\)\)*)
(*\!\(TraditionalForm\`\**)
(*StyleBox["H", "TI"] \((\**)
(*StyleBox["I", "TI"])\) == *)
(*\*FractionBox[\(1\), \(2  \[Pi]\[ScriptDotlessI]\)] *)
(*\*SubsuperscriptBox[\(\[Integral]\), \(-\[ScriptDotlessI]\[Epsilon] - \[Infinity]\), \(-\[ScriptDotlessI]\[Epsilon] + \[Infinity]\)]\**)
(*StyleBox["d", "TI"] \**)
(*StyleBox["y", "TI"] \**)
(*FractionBox[*)
(*RowBox[{*)
(*UnderoverscriptBox["\[Product]", *)
(*RowBox[{*)
(*StyleBox["k", "TI"], "==", "1"}], "4",*)
(*LimitsPositioning->True], *)
(*SuperscriptBox[*)
(*RowBox[{"(", *)
(*RowBox[{"1", "-", *)
(*StyleBox["y", "TI"], *)
(*SubscriptBox[*)
(*StyleBox["I", "TI"], *)
(*StyleBox["k", "TI"]]}], ")"}], *)
(*RowBox[{*)
(*RowBox[{"-", "1"}], "/", "2"}]]}], *)
(*StyleBox["y", "TI"]]\)*)
(**)


(* ::Input::Initialization:: *)
ClearAll[H,W];


(* ::Input::Initialization:: *)
H[II_]/;VectorQ[II,NumericQ]:=
Block[{S,Q, y,shift,lowerRoots},
lowerRoots = Select[Im[1/II],Negative];
If[lowerRoots =={}, Return [0]];
(*Print["lowerRoots",lowerRoots];*)
shift =0.75  Max[lowerRoots];
(*Print[shift];*)
S =1/ Sqrt[Times @@(1 - (y + I shift)  II)];
1/(2Pi I ) NIntegrate[ S/(y + I shift),{y,-Infinity,Infinity},
PrecisionGoal->12,WorkingPrecision->16, Method->"GaussKronrodRule", MaxRecursion->30]
];


(* ::Input::Initialization:: *)
W[RR_]/;VectorQ[RR,NumericQ]:=
Block[{Sing, x, shift,uppeRoots, h},
upperRoots = Select[Im[RR],Positive];
If[uppeRoots =={}, Return [0]];
shift =1/4  Min[upperRoots];
(*Print[shift];*)
h= -shift + Log[With[{II =Im[RR]/(x+ I shift-  RR)},Quiet[H[ II]]]/Sqrt[Times@@(x+ I shift-RR)]/(2 Pi) ] ;
NIntegrate[Exp[I x +h],{x,-Infinity,Infinity}]
]//Quiet;


(* ::Input:: *)
(**)


(* ::Input:: *)
(*RR1 = Table[Random[Complex] ,4]*)


(* ::Input:: *)
(*With[{II =Im[RR1]/(1-  RR1)},H[ II]]*)


(* ::Input:: *)
(*0.81846693698820914025001858879330708323`16. -0.59829776184813498623081825824460933359`16. I*)


(* ::Input:: *)
(*0.80925058287522027912335272607388711897`16. -0.36947736816569238814962818733504140152`16. I*)


(* ::Input:: *)
(*W[RR1]*)
