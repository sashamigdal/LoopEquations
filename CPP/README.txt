How to build:
1. $ module load gcc/9.2.0
2. $ module load cmake/3.18.4
3. $ module load cuda/11.2.67

4. $ cmake . -D CMAKE_BUILD_TYPE=Release -B cmake-build-release
5. $ cmake --build cmake-build-release/
OR
4. $ cmake . -D CMAKE_BUILD_TYPE=Debug -B cmake-build-debug
5. $ cmake --build cmake-build-debug/

Important: `ARPACK++` conflicts with `MKL` library. That means `numpy` and `scipy` must be installed
 using `pip`, not `conda`.

Benchmark:
params `-CPU 1 -NLAM 0 -Mu 1e-7 --serial` with T=1000
On Max's VM: 1.76e7 random walk steps per second (on CPU).
On Jubail CPU node (preempt): 1.77e7.
