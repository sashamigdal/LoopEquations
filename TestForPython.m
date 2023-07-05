(* ::Package:: *)

(* ::Input::Initialization:: *)
Get["/Users/am10485/Documents/Wolfram Mathematica/PadeBorel.m"];
pb[t_]:=PadeBorel[t , 1+Sum[x^n n!,{n,1,12}],x , 5]
