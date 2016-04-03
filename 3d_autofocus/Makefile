all: cpp cuda

cpp: grad_h.o
	clang++ -std=c++11 -O3 -Wall -Wextra -pedantic -lpthread grad_h.o grad_h_driver.cpp -o grad_h_driver

cuda: grad_h_cuda.o
	clang++ -std=c++11 -O3 -Wall -Wextra -pedantic -L/usr/local/cuda-7.5/lib64/stubs -L/usr/local/cuda-7.5/lib64 -I/usr/local/cuda-7.5/include -lcuda -lcudart -lcublas grad_h_cuda.o grad_h_driver.cpp -o grad_h_driver_cuda

grad_h.o:
	clang++ -std=c++11 -O3 -Wall -Wextra -pedantic -fPIC -c grad_h.cpp -o grad_h.o

grad_h_cuda.o:
	/usr/local/cuda/bin/nvcc -O3 -std=c++11 -Xcompiler -fPIC -c grad_h.cu -o grad_h_cuda.o

clean:
	rm -f *.o grad_h_driver grad_h_driver_cuda
