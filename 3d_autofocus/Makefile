MATLAB=/usr/cots/mathworks/2014a/bin
CXXFLAGS=-std=c++11 -O3 -fPIC -Wall -Wextra

LDFLAGS=-L"/usr/local/cuda-7.5/lib64/stubs" -L"/usr/local/cuda-7.5/lib64" \
-lcuda -lcudart -lcublas

INCLUDE=-I"/usr/local/cuda-7.5/include" \
-I"/usr/cots/mathworks/2014a/extern/include" \
-I"/usr/cots/mathworks/2014a/simulink/include"

all: driver cpp

driver: grad_h.o
	clang++ $(CXXFLAGS) -lpthread grad_h.o grad_h_driver.cpp -o grad_h_driver

driver_cuda: grad_h_cuda.o
	clang++ $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) grad_h_cuda.o grad_h_driver.cpp -o \
	grad_h_driver_cuda

cpp: grad_h.o
	$(MATLAB)/mex CXXFLAGS="$(CXXFLAGS)"  grad_h.o grad_h_mex.cpp

cuda: grad_h_cuda.o
	$(MATLAB)/mex CXXFLAGS="$(CXXFLAGS)" LDFLAGS="$(LDFLAGS)" \
	INCLUDE="$(INCLUDE)" grad_h_cuda.o grad_h_mex.cpp

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

grad_h.o:
	clang++ -std=c++11 -O3 -Wall -Wextra -pedantic -fPIC -c grad_h.cpp -o \
	grad_h.o

grad_h_cuda.o:
	/usr/local/cuda/bin/nvcc -O3 -std=c++11 -Xcompiler -fPIC -DNDEBUG -c \
	grad_h.cu -o grad_h_cuda.o

clean:
	rm -f *.o grad_h_driver grad_h_driver_cuda
