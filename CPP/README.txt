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