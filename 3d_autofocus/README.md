# 3D Autofocus

## Build

There are three build commands. 'make' will compile the C++ and CUDA
implementations of 'gradH' and build two executables 'grad\_h\_driver' and
'grad\_h\_driver\_cuda', respectively, which are driver programs for each
implementation. The following 'mex' commands build the MEX version of 'gradH'
linked against the C++ and CUDA implementations, respectively.

1) `make`
2) `mex CXXFLAGS="-std=c++11 -O3 -fPIC"                                                                                                                                 grad_h.o grad_h_mex.cpp`
3) `mex CXXFLAGS="-std=c++11 -O3 -fPIC  -I/usr/local/cuda-7.5/include" LDFLAGS="-L/usr/local/cuda-7.5/lib64/stubs -L/usr/local/cuda-7.5/lib64 -lcuda -lcudart -lcublas" grad_h_cuda.o grad_h_mex.cpp`
