#include <stdio.h>
#define N 100
__global__
void add(int*a,int*b){
int i=blockIdx.x;
if(i<N){b[i]=2*a[i];}
}
int main(){
int ha[N],hb[N];
int *da,*db;
cudaMalloc((void**)&da,N*sizeof(int));
cudaMalloc((void**)&db,N*sizeof(int));
for(int i=0;i<N;i++){ha[i]=i;}
cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    add<<<N, 1>>>(da, db);
    //
    // Copy output array from GPU back to CPU.
    //
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }//
    // Free up the arrays on the GPU.
    //
    cudaFree(da);
    cudaFree(db);
    return 0;
}