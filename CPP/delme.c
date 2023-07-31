#include <math.h>
int main() {
    double snm=0;
    int lim = 10000;
#pragma omp parallel for reduction (+:snm)
    for(int i =0; i <1000000000; i++){
        snm +=  sin(i);
    }
    return snm;
}
