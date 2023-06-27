(* ::Package:: *)

(* ::Input::Initialization:: *)
(* Pade-Borel approximation of the asymptotic series\
Sum[ fn x^n] = Integrate[E^(-t)Sum[ fn (t x)^n/n!],{t,0,Infinity}]*)
test = PadeApproximant[Sum [(n+3)/(n+1)x^n,{n,0,100}],{x,0,{2,2}}]


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
(3-(41 x)/20+(11 x^2)/120)/(1-(27 x)/20+(3 x^2)/8)


(* ::Output::Initialization:: *)
1/(1-x)


(* ::Output::Initialization:: *)
1/(1-x)


(* ::Output::Initialization:: *)
1/(1-x)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Output::Initialization:: *)
(1+(5 t x)/4+(5 t^2 x^2)/16)/(1+(3 t x)/4+(t^2 x^2)/16)


(* ::Input::Initialization:: *)
(* bad method *)


(* ::Input::Initialization:: *)
(*Assuming[Re[t] >0,Integrate[test Exp[-x],{x,0,Infinity}]]*)


(* ::Input::Initialization:: *)
II[z_,t_] =Assuming[{Re[z/t]<0},Integrate[Exp[-y]/(y t-z),{y,0,Infinity}]]//FullSimplify


(* ::Output::Initialization:: *)
(E^(-(z/t)) (-CoshIntegral[z/t]+Log[-(t/z)]+Log[z/t]-SinhIntegral[z/t]))/t


(* ::Output::Initialization:: *)
(E^(-(z/t)) (-CoshIntegral[z/t]+Log[-(t/z)]+Log[z/t]-SinhIntegral[z/t]))/t


(* ::Output::Initialization:: *)
(E^(-(z/t)) (-CoshIntegral[z/t]+Log[-(t/z)]+Log[z/t]-SinhIntegral[z/t]))/t


(* ::Output::Initialization:: *)
(E^(-(z/t)) (-CoshIntegral[z/t]+Log[-(t/z)]+Log[z/t]-SinhIntegral[z/t]))/t


(* ::Output::Initialization:: *)
(E^(-(z/t)) (-CoshIntegral[z/t]+Log[-(t/z)]+Log[z/t]-SinhIntegral[z/t]))/t


(* ::Output::Initialization:: *)
(E^(-(z/t)) (-CoshIntegral[z/t]+Log[-(t/z)]+Log[z/t]-SinhIntegral[z/t]))/t


(* ::Input::Initialization:: *)
II[z_, t_, 1] := II[z,t];
II[z_, t_, n_] := D[II[z,t],{z,n-1}]/(n-1)!/; n \[Epsilon] PositiveIntegers;


(* ::Input::Initialization:: *)
MyResidue[R_,{x_,x0_,n_}]:=n!/D[1/R,{x,n}]/.x->x0


(* ::Input::Initialization:: *)
PadeBorel[t_, F_,x_, M_]/; PolynomialQ[F,x] :=
Block[{G,res, M2 = Evaluate[2 M]},
G =Evaluate[N[PadeApproximant[Evaluate[F/.{x^n_ :> x^n/n!}],{x,0,{M,M}}],20]];
(*Print["G=",G];*)
If[t == 0, Return[G/.x->0]];
fp =FunctionPoles[G,x];
(*Print["fp=", N[fp]];*)
res = Plus @@N[MyResidue[G,{x,#[[1]],#[[2]]}] Evaluate[II[#[[1]],t,#[[2]]]]&/@fp];
If[NumberQ[t],res, FullSimplify[res]]
];


(* ::Input::Initialization:: *)
PadeBorel[t, 1 + Sum[Binomial[1/2,k] x^k,{k,1,10}],x , 3]


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -2.52323 I/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +2.52323 I/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -2.52323 I/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +2.52323 I/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -2.52323 I/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +2.52323 I/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -2.52323 I/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +2.52323 I/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
1/t ((9.756 +18.6514 I) E^(7.11536 -(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 -2.52323 I)/t)]+Log[-((7.11536 -2.52323 I)/t)]+Log[(0.124842 +0.0442711 I) t]-1. SinhIntegral[-((7.11536 -2.52323 I)/t)])+(9.756 -18.6514 I) E^(7.11536 +(0. +2.52323 I)/t) (-1. CoshIntegral[-((7.11536 +2.52323 I)/t)]+Log[-((7.11536 +2.52323 I)/t)]+Log[(0.124842 -0.0442711 I) t]+SinhIntegral[(7.11536 +2.52323 I)/t])-298.617 E^(29.0634/t) (-1. CoshIntegral[-(29.0634/t)]+Log[-(1/t)]+Log[t]+SinhIntegral[29.0634/t]))


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Output::Initialization:: *)
0.867262 +0. I


(* ::Input::Initialization:: *)
PadeBorel[t , 1+Sum[x^n n!,{n,1,20}],x , 7]


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1./t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1./t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1/t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1/t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1/t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1/t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1/t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1/t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1/t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1/t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1/t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1/t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1/t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1/t]))/t)


(* ::Output::Initialization:: *)
-((1. E^(-1./t) (-1. CoshIntegral[1/t]+Log[1/t]+Log[-1. t]-1. SinhIntegral[1/t]))/t)


(* ::Input:: *)
(**)


(* ::Input::Initialization:: *)
ClearAll[BP];


(* ::Input::Initialization:: *)
PadeBorel[tmin_, tmax_, steps_, F_ ,x_, M_]:=
Block[{PB,MyTable, t, dt= (tmax-tmin)/steps},
PB[t_] := If[t==0,Evaluate[F/.x->0],PadeBorel[t,F,x,M]];
MyTable =If[steps >= 1000, ParallelTable, Table];
MyTable[PB[t],{t, tmin, tmax, dt}]
]



(*PadeBorel[-10., 10., 1000,1+Sum[x^n n!,{n,1,12}],x, 5];*)


(* ::Input:: *)
(*PadeBorel[1, 1 + Sum[Binomial[1/2,k] x^k,{k,1,10}],x , 4]*)
