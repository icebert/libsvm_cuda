#ifndef _CUDA_LIBSVM_H
#define _CUDA_LIBSVM_H


#define DEVICE_HEAP_SIZE 3*1024*1024*1024L



int cuda_svm_train(const struct svm_problem *prob, struct svm_problem *subprobs, struct svm_parameter *params, int nr_grid, int nr_fold, struct svm_model *submodels);


#endif /* _CUDA_LIBSVM_H */

