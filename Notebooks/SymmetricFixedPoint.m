(* ::Package:: *)

(* ::Input:: *)
(*Sqr[F_] := Expand[F . F]*)


(* ::Input:: *)
(*F[k_] := {u[k],v[k], I b} (* anzatz*)*)


(* ::Input:: *)
(*Eq3 =FullSimplify[Sqr[F[k+1] - F[k]]-1]*)


(* ::Input:: *)
(**)
(*(* My way around bugs in  ReIm[] *)*)


(* ::Input:: *)
(*MyReIm[Ex_] := ReIm[ComplexExpand[Ex]]/.{Im[X_]:>0, Re[X_] :>X}*)


(* ::Input:: *)
(*Eq4 =(Sqr[F[k+1]] - Sqr[F[k]] - I \[Gamma])^2 + \[Gamma]^2-Sqr[F[k+1] + F[k]]*)


(* ::Input:: *)
(*{Eq4R , Eq4I} =MyReIm[Eq4]//FullSimplify*)


(* ::Input:: *)
(*(* we see that Eq4I is satisfied if u^2 + v^2 does not depend on k *)*)


(* ::Input:: *)
(*Sub = {u[l_] :> r Cos[\[Alpha][l]], v[l_] :> r Sin[\[Alpha][l]]}*)


(* ::Input:: *)
(*Eq4I/.Sub//FullSimplify*)


(* ::Input:: *)
(*(* now we have to look for solutions for \[Alpha][k], r, b *)*)


(* ::Input:: *)
(*Eq5 =Eq4R/.Sub//FullSimplify*)


(* ::Input:: *)
(*Eq6 =Eq3 /.Sub//FullSimplify*)


(* ::Input:: *)
(*ClearAll[\[Beta]];*)


(* ::Input:: *)
(*rbsol =Simplify[Solve[{Eq5==0, Eq6==0}, {b, r}][[-1]]/.\[Alpha][k]-\[Alpha][1+k]->-\[Beta],{0 < \[Beta] < Pi}]//TrigReduce*)


(* ::Input:: *)
(*(* here is our symmetric solution *)*)


(* ::Input:: *)
(*ff[k_, \[Alpha]_, \[Beta]_] =FullSimplify[(F[k]/.Sub/.rbsol),0 < \[Beta] < Pi]*)


(* ::Input:: *)
(*(* compute vorticity vector *)*)


(* ::Input:: *)
(*Cross[ff[k,\[Alpha],\[Beta]],ff[k+1,\[Alpha],\[Beta]]]*)


(* ::Input:: *)
(*(* make it simpler *)*)


(* ::Input:: *)
(*FullSimplify[I Cross[ff[k,\[Alpha],\[Beta]],ff[k+1,\[Alpha],\[Beta]]]/.{\[Alpha][k+1]-> \[Alpha][k] + \[Sigma] \[Beta]}]/.{Sin[\[Sigma] \[Beta]]-> \[Sigma] Sin[\[Beta]], Sin[\[Sigma] \[Beta]/2]-> \[Sigma] Sin[\[Beta]/2]}//Simplify*)


(* ::Input:: *)
(*(* this is a lightlike vectotr (zero square))*)*)


(* ::Input:: *)
(*FullSimplify[Sqr[Cross[ff[k,\[Alpha],\[Beta]],ff[k+1,\[Alpha],\[Beta]]]]]//.{\[Alpha][k+1]-> \[Alpha][k] + \[Sigma][k] \[Beta], Cos[\[Beta] \[Sigma][k]]->Cos[\[Beta]]}*)


(* ::Input:: *)
(*(* the general formula for dissipation on the solution of recurrent equations*)*)


(* ::Input:: *)
(*Diss =Expand[Subscript[F, k] . Subscript[F, k] - 1/4(\[Sigma] Sqrt[4Subscript[F, k] . Subscript[F, k] + \[Gamma](2 I - \[Gamma])] - I \[Gamma])^2]/.\[Sigma]^2->1//FullSimplify*)


(* ::Input:: *)
(*1/2 \[Gamma] (\[Gamma]+I (-1+\[Sigma] Sqrt[-\[Gamma] (-2 I+\[Gamma])+4 Subscript[F, k] . Subscript[F, k]]))*)


(* ::Input:: *)
(*(* check how it vanishes on a symmetric soluion *)*)


(* ::Input:: *)
(*Test =FullSimplify[Diss//.{Subscript[F, k] . Subscript[F, k] ->1/4}]*)


(* ::Input:: *)
(*FullSimplify[Test/.Sqrt[-(-I+\[Gamma])^2]->I (\[Gamma] -I)]*)


(* ::Input:: *)
(*FullSimplify[ff[k+1,M] - ff[k,M]]*)


(* ::Input:: *)
(*(ff[k+1,M] - ff[k,M]) . (ff[k+1,M] - ff[k,M])//FullSimplify*)


(* ::Input:: *)
(*(* linearized discrete loop equations*)*)


(* ::Input:: *)
(*(*vector version of ReIm *)*)


(* ::Input:: *)
(*RI[X_] := Transpose[ComplexExpand[ReIm[X]]];*)


(* ::Input:: *)
(*RI[{a +I p,b+ I q,c+ I r}]*)


(* ::Input:: *)
(*Append[RI[(a +I p)^2 ==1],a==2]*)


(* ::Input:: *)
(*(* equations from my paper, for reference *)*)


(* ::Input:: *)
(*(* this formula for \[CapitalLambda] is not correctly displayed because of Mathematica Latex bugs*)*)


(* ::Input:: *)
(*(* formulas needed for solution for Re[\[Mu]_k]*)*)


(* ::Input:: *)
(*\[Gamma]formula[u_] := Block[{ V, Q},*)
(*V=ReIm[u];*)
(*Q = PseudoInverse[V . Transpose[V]] . V;*)
(*(#[[2]] + I #[[1]])&/@ Q*)
(*];*)
(*\[Gamma]null[u_] := Block[{ V},*)
(*V=ReIm[u];*)
(*NullSpace[V . Transpose[V]]*)
(*];*)
(**)


(* ::Input:: *)
(*(* Block matrix algebra *)*)


(* ::Input:: *)
(*{1,2}[[;;2]]*)


(* ::Input:: *)
(*ClearAll[MDot, VMDot, MVDot, VDot];*)
(*(*algebra of matrix operators and vector operators*)*)


(* ::Input:: *)
(*MOp[a_,b_, Op_] := *)
(*Block[{K,L,M},*)
(*K= Dimensions[a][[1]];*)
(*If[Length[Dimensions[b]]>=2,*)
(*L= Dimensions[b][[1]];*)
(*M= Dimensions[b][[2]]*)
(*,*)
(*L= Dimensions[a][[2]];*)
(*M= Dimensions[b][[1]]*)
(*];*)
(*ParallelTable[Total[Op[a[[k,#]],b[[#,m]]]&/@ Range[L]],{k,K},{m,M}]*)
(*];*)


(* ::Input:: *)
(*MDot[a_,b_] := MOp[a,b,Dot];*)
(*MProd[a_,b_] := MOp[a,b,TensorProduct];*)


(* ::Input:: *)
(*Flatten[{{x . a+y . b},{w . b+z . a}},2]*)


(* ::Input:: *)
(*{{a . x+b . y}} [[1,1]]*)


(* ::Input:: *)
(*MDot[{{a,b},{c,d}},{{x,y},{z,w}}]*)


(* ::Input:: *)
(*MProd[{{a,b},{c,d}},{{x,y},{z,w}}]*)


(* ::Input:: *)
(*MDot[{{a,b}},{{x,y},{z,w}}]*)


(* ::Input:: *)
(*MDot[{{x,y},{z,w}},{{a,b}}]*)


(* ::Input:: *)
(*MDot[{{x,y},{z,w}},Transpose[{{a,b}}]]*)


(* ::Input:: *)
(*MDot[{{a,b}},Transpose[{{x,y}}]]*)


(* ::Input:: *)
(**)


(* ::Input:: *)
(*MDot[{{a,b},{c,d}},{{x,y},{z,w}}]*)


(* ::Input:: *)
(*VMDot[v_,m_] :=  MDot[{v},m][[1]];*)


(* ::Input:: *)
(*MVDot[m_,v_] := First/@MDot[m,Transpose[{v}]];*)


(* ::Input:: *)
(*VDot[u_, v_] :=  MDot[{u},Transpose[{v}]][[1,1]];*)


(* ::Input:: *)
(*MVDot[{{x,y},{z,w}},{a,b}]*)


(* ::Input:: *)
(*VMDot[{a,b},{{x,y},{z,w}}]*)


(* ::Input:: *)
(*VDot[{a,b},{x,y}]*)


(* ::Input:: *)
(*ClearAll[i,j,k,l];*)


(* ::Input:: *)
(*FromBlock[A_]:=ArrayFlatten[A];*)


(* ::Input:: *)
(*ToBlock[B_]:=Partition[B,#/2& /@ Dimensions[B]];*)


(* ::Input:: *)
(**)
(*PseudoInverseFromBlock[B_]:=*)
(*Block[{A, M},*)
(*M = Length[B[[1,1]]];*)
(*A = FromBlock[B];*)
(*S = {#[[;;M]],#[[M+1;;]]}& /@NullSpace[A];*)
(*{ToBlock[PseudoInverse[A]],S}*)
(*]*)


(* ::Input:: *)
(*(* testing block matrix ops *)*)


(* ::Input:: *)
(*Block[{a,b,c,d, MD,M1, M2,A, B, C, D, NS},*)
(*a = {{1,2,1},{5,6,1},{1,2,1}};b = {{3,4,1},{7,8,1},{3,4,1}};*)
(*c = {{9,10,1},{13,14,1},{9,10,1}};d = {{11,12,1},{15,16,1},{11,12,1}};*)
(*M1 ={{a[[1]],b[[1]]},{c[[2]],d[[2]]}};*)
(*M2 = {{c[[1]],d[[3]]},{a[[2]],b[[2]]}};*)
(*Print["M1=",MatrixForm[M1]];*)
(*Print["M2=",MatrixForm[M2]];*)
(*Print["M1-M2=",MatrixForm[M1-M2]];*)
(*MD =MDot[M1,M2];*)
(*Print["M1.M2=",MatrixForm[MD]];*)
(*Print[VMDot[{a[[1]],c[[2]]},M1]];*)
(*Print[MatrixForm[{{a,b},{c,d}}]];*)
(*A = FromBlock[{{a,b},{c,d}}];*)
(*Print["FromBlock :",MatrixForm[A]];*)
(*B = ToBlock[A];*)
(*Print[MatrixForm[B]];*)
(*{C ,NS}= PseudoInverseFromBlock[B];*)
(*(*NS = #[[1]]- I #[[2]]& /@ NS;*)*)
(*Print["NS=",NS];*)
(*Print[MatrixForm[C]];*)
(*{D,NS} = PseudoInverseFromBlock[C];*)
(*Print["NS=",NS];*)
(*Print[MatrixForm[D]];*)
(*]*)


(* ::Input:: *)
(*(* *)
(*the linearized algorithm from the paper, *)
(*with MyTable = Table or ParallelTable*)
(* (for large order M)*)
(**)*)


(* ::Input:: *)
(*Thread[{{1,2},{0,1}}-{{3,4},{1,0}},{1}]*)


(* ::Input:: *)
(*ConstantArray[0,{2,2}]& /@ Range[3]*)


(* ::Input:: *)
(**)
(*SDEMatrices[F_,MyTable_, Digits_] :=*)
(*Block[{ *)
(*M,K,q,qd,G,U,V,\[Gamma],*)
(*LL, RR, MM,PP,QQ,TT,*)
(*P,Q,PQM,PQR,NS,\[CapitalLambda],IM, I3,Z3,X,Y,Z,\[CapitalTheta],CC, CCI,*)
(*i,j,k,n*)
(*},*)
(*M = Length[F];*)
(*IM = ConstantArray[I,{M}];*)
(*q = Thread[RotateRight[F,1]- F,1];*)
(*(*Return[MatrixForm[q]];*)*)
(*qd = Normal @ HodgeDual[#] &/@ q;*)
(*(*Return[MatrixForm[qd]];*)*)
(*G =2 F[[#]]q[[#]]-I &/@ Range[M];*)
(*(*Return[G];*)*)
(*(* G is a complex M-array *);*)
(*U =-G[[#]] F[[#]] . qd[[#]] &/@ Range[M];*)
(*(* U is a  M-array of complex 3 vectors*);*)
(*(*Return[U];*)*)
(*\[Gamma] =\[Gamma]formula/@ U;*)
(*(*If[Length[Flatten[\[Gamma]null/@ U]] >0, Print["Nullspace!"]];*)*)
(*(*Return[\[Gamma]];*)*)
(*V =2 G[[#]]*q[[#]]-F[[#]] &/@ Range[M];*)
(*(* V is a  M-array of complex 3 vectors*);*)
(*(*Return[V];*)*)
(*(* the arrays  of matrices *) *)
(*Z3 =ConstantArray[0,{3,3}];*)
(*LL = MyTable[If[j< i,\[Gamma][[i]]\[TensorProduct]V[[i]] . qd[[j]],Z3],{i,M},{j,M}];*)
(*(*Print["LL:", Dimensions[LL]];*)*)
(*X = Re[LL];*)
(*X[[#,#]] += Re[\[Gamma][[#]]\[TensorProduct]U[[#]]]& /@ Range[M];*)
(*(*Print["X:", Dimensions[X]];*)
(*Print["LL:", Dimensions[LL]];*)*)
(*(*Return [X];*)*)
(*(*Print["MDot[Im[LL],X]:"];*)
(*Return[MDot[Im[LL],X]];*)*)
(*MM = X;*)
(*(*Print["MM:", Dimensions[MM]];*)*)
(*Do[MM = X - MDot[Im[LL],MM],M-1];*)
(*(* The arrays of vectors *)*)
(*(*Return[MM];*)*)
(*Y =\[Gamma][[#]]\[TensorProduct]V[[#]] & /@ Range[M];*)
(*(*M array of 3 X 3 complex matrix *)*)
(*(*Return[Y];*)*)
(*PP = Re[Y];*)
(*QQ = -Im[Y];*)
(*Do[PP =Re[Y] +MVDot[ Im[LL],PP],M-1];*)
(*Do[QQ =-Im[Y] +MVDot[ Im[LL],QQ],M-1];*)
(*(*Return[PP - I QQ];*)*)
(*(*Return[PP];*)*)
(*P= VDot[qd,PP];*)
(*(*3 X 3 complex matrix *)*)
(*(*Return[P];*)*)
(*(*Print["P, Q = ",{MatrixForm[P], MatrixForm[Q]}];*)*)
(*Q=VDot[qd,PP]; (*Total[qd[[#]].QQ[[#]]& /@ Range[M]];*)*)
(*(*3 X 3 complex matrix *)*)
(**)
(*(*Return[MatrixForm[{ {Re[P],Re[Q]}, {Im[P], Im[Q]}}]];*)*)
(**)
(*{PQM,NS} =PseudoInverseFromBlock[{ {Re[P],Re[Q]}, {Im[P], Im[Q]}}];*)
(*(* PQM is 2 X 2 block matrix with 3X3 real matrix elements *)*)
(*(*Return[MatrixForm[PQM]];*)*)
(*(* now NS is an array  with 3X3 real matrix elements *)*)
(*(*Print[NS];*)*)
(*NS = (#[[1]]- I #[[2]])&/@NS;*)
(*(* now NS is an array  with complex 3-vector elements *)*)
(*(*Return[NS];*)*)
(*K = Length[NS];*)
(*(*Return[NS];*)*)
(*I3 = IdentityMatrix[3];*)
(*X =VDot[VMDot[{I3, I I3},PQM],{I3, - I I3}];*)
(*(* X is a 3X3 complex matrix *)*)
(*(*Return[X];*)*)
(*(*Return[NS];*)*)
(*(* Y is a M vector with elements being complex 3X3 matrices*)*)
(*Y =Re[X . #]&/@ qd;*)
(*(* MM is a M by M matrix with elements being real 3X3 matrices*)*)
(*(* Lambda is a M vector with elements being 3X3 complex matrices*)*)
(*\[CapitalLambda] =-(Im[ X . #]&/@ qd) -VMDot[Y,MM];*)
(*(*Return[\[CapitalLambda]];*)*)
(*(* Z is a K by M matrix with elements being complex 3 vectors*)*)
(*Z= MyTable[ NS[[i]] . qd[[n]],{i,K},{n,M}];*)
(*(*Return[Z];*)*)
(*(* Theta is a K by M matrix with elements being real 3 vectors*)*)
(*\[CapitalTheta]=Im[Z] - MDot[Re[Z],MM];*)
(*(*Return[\[CapitalTheta]];*)*)
(*CC =MDot[\[CapitalTheta],Transpose[\[CapitalTheta]]];*)
(*(*Return[MatrixForm[CC]];*)*)
(*(* CC is a K by K symmetric real positive matrix*)*)
(*Print["eigenvalues=",Eigenvalues[CC]];*)
(*CCI = Inverse[CC];*)
(*(*Print["CC=", MatrixForm[CC]];*)
(*Print["CCI=",MatrixForm[CCI]];*)
(*Print["CCI.CC=",N[MatrixForm[CCI.CC],Digits]];*)*)
(*Y =CCI . \[CapitalTheta];*)
(*(* Y is a K by M  array of real 3-vectors same as theta*)*)
(*(*Return[MatrixForm[MyTable[Sum[Y[[i,j]].\[CapitalTheta][[k,j]],{j,M}],{i,K},{k,K}]]];*)*)
(*(*Return[Y];*)*)
(*(*test =Map[CC.# & , Y,{2}];*)
(*Return[{MatrixForm[\[CapitalTheta]],MatrixForm[test]}];*)*)
(*X = VMDot[\[CapitalLambda],Transpose[\[CapitalTheta]]];*)
(*(*Return[X];*)*)
(*(* X is a K  array of real 3-vectors*)*)
(*(*Print["X= ", X];*)
(*Print["Y= ", Y];*)*)
(*(*Print["VMDot[test,Y]:",test];*)*)
(*\[CapitalLambda][[#]]-=Sum[X[[k]]\[TensorProduct]Y[[k,#]],{k,K}] & /@ Range[M];*)
(*(*Print[\[CapitalLambda]];*)*)
(*(*Return[\[CapitalLambda]];*)*)
(*(*X =VMDot[\[CapitalLambda],Transpose[\[CapitalTheta]]];*)
(*Return[MatrixForm[X]];*)*)
(*X = MDot[MM,Transpose[\[CapitalTheta]]];*)
(*(* X is a MXK  array of real 3-vectors*)*)
(*(* Y is a K X M  array of real 3-vectors same as theta*)*)
(*(*TT = MyTable[MM[[i,j]] - Total[X[[i,#]]\[TensorProduct]Y[[#,j]]& /@Range[K]],{i,M},{j,M}];*)*)
(*TT = MM-MProd[X,Y];*)
(*(*MyTable[MM[[i,j]] - Total[X[[i,#]]\[TensorProduct]Y[[#,j]]& /@Range[K]],{i,M},{j,M}];*)*)
(*(* TT is a MXM  array of real 3X3-matrices*)*)
(*TT[[1]] += \[CapitalLambda];*)
(*Do[TT[[k]] += TT[[k-1]],{k,2,M}];*)
(*ArrayFlatten[TT]*)
(*];*)


(* ::Input:: *)
(*SDE[M_, MyTable_, Digits_, \[Sigma]_,Reps_] :=*)
(*Block[{O,M3, X,f0,\[Alpha]0,\[Beta]0, X0,F,Vars,Varst,A,B,W, proc, t, dist},*)
(*O= RandomVariate[CircularRealMatrixDistribution[3]];*)
(*(*O = IdentityMatrix[3];*)*)
(*(* The arrays of vectors *)*)
(*\[Beta]0 =2 Pi/M;*)
(*f0[k_] := N[ff[k,# \[Beta]0&,\[Beta]0]];*)
(*X0 = Flatten[f0/@ Range[M]];*)
(*(*Print[X0];*)*)
(*M3 = Length[X0];*)
(*A = ConstantArray[0.,M3];*)
(*If[Reps ==0,*)
(*B[X_] := IdentityMatrix[M3] +\[Sigma] TensorProduct[X,X];*)
(*X =Table[Unique[],M3];*)
(*proc =ItoProcess[{A,B[X],X},{X,X0},t];*)
(*RandomFunction[proc,{0.,5.,0.01}],*)
(*B[X_]:=SDEMatrices[Partition[X,{3}],MyTable,Digits];*)
(*X = X0;*)
(*dist = NormalDistribution[0,\[Sigma]];*)
(*Do[*)
(*W = Table[RandomVariate[dist],M3];*)
(*A = B[X];*)
(*X += A . W;*)
(*(*Print[{(t+1)\[Sigma], Partition[X,{3}]}];*)*)
(*,{t,Reps}];*)
(*];*)
(*];*)


(* ::Input:: *)
(*(* It runs fine for a sample B[X_] matrix (case Reps=0), but hangs for a real one. On the other hand, direct iterations with real B[X_] and Gaussian noise run just fine (case Reps = 10). Something is wrong with IroProcess *)*)


(* ::Input:: *)
(*SDE[30,ParallelTable, 14,0.001, 10];*)



