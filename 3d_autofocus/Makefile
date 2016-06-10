MATLAB=/usr/cots/mathworks/2014a/bin
CXXFLAGS=-std=c++11 -O3 -fPIC -Wall -Wextra

LDFLAGS=-L"/usr/local/cuda-7.5/lib64/stubs" -L"/usr/local/cuda-7.5/lib64" \
-lcuda -lcudart -lcublas

INCLUDE=-I"/usr/local/cuda-7.5/include" \
-I"/usr/cots/mathworks/2014a/extern/include" \
-I"/usr/cots/mathworks/2014a/simulink/include"

cpp: minimize_entropy.o grad_h.o entropy.o
	$(MATLAB)/mex CXXFLAGS="$(CXXFLAGS)" entropy.o grad_h.o minimize_entropy.o \
		minimize_entropy_mex.cpp

cuda: minimize_entropy.o grad_h_cuda.o entropy.o
	$(MATLAB)/mex CXXFLAGS="$(CXXFLAGS)" LDFLAGS="$(LDFLAGS)" \
	INCLUDE="$(INCLUDE)" entropy.o grad_h_cuda.o minimize_entropy.o \
	minimize_entropy_mex.cpp

# -----------------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------------

grad_h_driver: grad_h.o
	clang++ $(CXXFLAGS) -lpthread grad_h.o grad_h_driver.cpp -o grad_h_driver

minimize_driver: minimize_entropy.o grad_h.o
	clang++ $(CXXFLAGS) -lpthread minimize_entropy.o grad_h.o \
		minimize_entropy_driver.cpp -o minimize_entropy_driver

grad_h_driver_cuda: grad_h_cuda.o
	clang++ $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) grad_h_cuda.o grad_h_driver.cpp -o \
	grad_h_driver_cuda

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

grad_h.o:
	clang++ -std=c++11 -O3 -Wall -Wextra -pedantic -fPIC -c grad_h.cpp -o \
	grad_h.o

minimize_entropy.o:
	clang++ -std=c++11 -O3 -Wall -Wextra -pedantic -fPIC -c \
		minimize_entropy.cpp -o minimize_entropy.o

entropy.o:
	clang++ -std=c++11 -O3 -Wall -Wextra -pedantic -fPIC -c entropy.cpp -o \
	entropy.o

grad_h_cuda.o:
	/usr/local/cuda/bin/nvcc -O3 -std=c++11 -Xcompiler -fPIC -DNDEBUG -c \
	grad_h.cu -o grad_h_cuda.o

clean:
	rm -f *.o grad_h_driver grad_h_driver_cuda *.mex*
