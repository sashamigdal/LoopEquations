#define _USE_MATH_DEFINES

#include "Eigen/Core"
#include <iostream>
#include <math.h>

std::complex<double> operator"" _i(long double x ) 
{ 
    return std::complex<double>(0.0, x);
} 

int main()
{
    double x [3] = {1, 2, 3};
    double y [3] = {4, 5, 7};

    double px = 0.5;
    double py = 0.5;

    Eigen::RowVectorXd m = Eigen::RowVectorXd::LinSpaced(3, -1, 1);

    Eigen::MatrixXcd v(3,3);
    v << (1.0 + 1.0_i), (2.0 + 1.0_i), (3.0 + 1.0_i),
         (1.0 + 2.0_i), (2.0 + 2.0_i), (3.0 + 2.0_i),
         (1.0 + 3.0_i), (2.0 + 3.0_i), (3.0 + 3.0_i);

    double Tr5 = ((2.0_i * M_PI / px * x[0] * m).array().exp().matrix() * v * (2.0_i * M_PI / py * y[0] * m).array().exp().matrix().transpose()).real()(0);

    std::cout << "Tr5 = " << Tr5 << std::endl;

    return 0;
}