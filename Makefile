CXX ?= g++
NVCC = nvcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -fpermissive -DCV_OMP 
NFLAGS = -ccbin $(CXX) -Xcompiler -Wall,-Wconversion,-fPIC,-fopenmp,-fpermissive,-DCV_OMP 
SHVER = 2
OS = $(shell uname)

ifeq ($(NVML_LIB),)
NVML_FLAGS = 
else
NVML_FLAGS = -lnvidia-ml -L $(NVML_LIB) -Xcompiler -DNVML
endif

# Gencode arguments
SMS ?= 30 35 37 50 52

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif


all: svm-train-gpu svm-predict svm-scale

lib: svm.o cuda_svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $(CFLAGS) $${SHARED_LIB_FLAG} svm.o cuda_svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o cuda_svm.o
	$(NVCC) $(NFLAGS) $(GENCODE_FLAGS) $(NVML_FLAGS) svm-predict.c svm.o cuda_svm.o -o svm-predict -lm
svm-train-gpu: svm-train.o svm.o cuda_svm.o
	$(NVCC) $(NFLAGS) $(GENCODE_FLAGS) $(NVML_FLAGS) svm-train.o svm.o cuda_svm.o -o svm-train-gpu -lm
svm-train.o: svm-train.c
	$(CXX) $(CFLAGS) -c svm-train.c
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
cuda_svm.o: cuda_svm.cu cuda_svm.h
	$(NVCC) $(NFLAGS) $(GENCODE_FLAGS) $(NVML_FLAGS) -c cuda_svm.cu
clean:
	rm -f *~ svm.o cuda_svm.o svm-train.o svm-train-gpu svm-predict svm-scale libsvm.so.$(SHVER)

