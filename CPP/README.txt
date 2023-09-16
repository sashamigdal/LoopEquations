How to build:
$ module load gcc/9.2.0
$ module load cmake/3.18.4
$ cmake . -D CMAKE_BUILD_TYPE=Release -B cmake-build-release
$ cmake --build cmake-build-release/
