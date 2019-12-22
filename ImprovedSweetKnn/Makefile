NVCC=nvcc -G --compiler-options -Wunused-parameter -arch=sm_35 -std=c++11 -pg -lcublas -O3 -Xcompiler -fopenmp

all: knnjoin
knnjoin: ./src/knnjoin.cu
	$(NVCC) -o $@ $^
clean:
	rm knnjoin *.out
