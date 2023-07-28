
#include <math.h>

#define __int64 int

class DPoint {
    double x,y;
public:
	typedef double T;
	typedef DPoint F;

	DPoint(){	// fields automatically set to zero
        x=y=0;
	}
	DPoint(T a, T b) {
		x=a,y=b;
	}
	
	double Dot(const & DPoint b) const {
		return x*b.x + y*b.y;
	}
	double Sum() const {
		return x + y;
	}
	double Dif() const {
		return y - x;
	}
	double Sqr() const {
		return x*x + y*y;
	}
	
	double Len() {
		return sqrt(Sqr());
	}
    DPoint operator +(const F &p) const{ 
		return DPoint(x + p.x, y + p.y); 
	}
    void operator +=(const F &p) { 
		x += p.x; y += p.y; 
	}
	void operator -=(const F &p) { 
		x -= p.x; y -= p.y; 
	}
	void operator *=(T f) { 
		x *= f; y *= f; 
	}
    void operator /=(T f) { 
		x /= f; y /= f; 
	}
};

DPoint F( int sigma, double beta){
    double f = 1 / (2 * sin(beta / 2)) , alpha = sigma * beta;
    return DPoint(f *cos(alpha), f* sin(alpha));
}

// '''
// def SS(n, m, M, sigmas, beta):
//     Snm = np.sum(F(sigmas[n:m], beta), axis=1)
//     Smn = np.sum(F(sigmas[m:M], beta), axis=1) + np.sum(F(sigmas[0:n], beta), axis=1)
//     snm = Snm / (m - n)
//     smn = Smn / (n + M - m)
//     ds = snm - smn
//     return np.sqrt(ds.dot(ds))
// '''

double DS(int n, int m, int M, double * sigmas, double beta) 
{
    DPoint smn, snm;
	if(M < 10000){
        for(int i =n; i <m; i++){
			snm += F(sigmas[i], beta);
		}
        for(int i =0; i <n; i++){
			smn += F(sigmas[i], beta);
		}
        for(int i =m; i <M; i++){
			snm += F(sigmas[i], beta);
		}
	}else{
        #pragma omp parallel for reduction (+:snm)
            for(int i =n; i <m; i++){
                snm +=  F(sigmas[i], beta);
            }
        #pragma omp parallel for reduction (+:smn)
            for(int i =0; i <n; i++){
                smn +=  F(sigmas[i], beta);
            }
        #pragma omp parallel for reduction (+:snm) 
            for(int i =m; i <M; i++){
                smn +=  smn + F(sigmas[i], beta);
            }
    }
    snm /= double(m-n);
    smn /= double(n + M - m);
    smn -= snm;
	return smn.Len();
}

