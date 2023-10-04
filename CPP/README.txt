How to build:
1. $ module load gcc/9.2.0
2. $ module load cmake/3.18.4
3. $ cmake . -D CMAKE_BUILD_TYPE=Release -B cmake-build-release
4. $ cmake --build cmake-build-release/
OR
3. $ cmake . -D CMAKE_BUILD_TYPE=Debug -B cmake-build-debug
4. $ cmake --build cmake-build-debug/

Important: `ARPACK++` conflicts with `MKL` library. That means `numpy` and `scipy` must be installed
 using `pip`, not `conda`.

Benchmark:
On Max's VM the params `-CPU 1 -NLAM 0 -Mu 1e-7 --serial` with T=1000 give 1.76e7 random walk steps per second (on CPU).