/**
 * @file functions.h
 * @author luwangguerde@163.com
 * @brief Functions
 * @version 0.1
 * @date 2024-11-05
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "base.h"

typedef struct VEC Label;   // one-hot use intArray
typedef struct VEC Output;  // model-output use doubleArray
typedef struct VEC Input;   // model-input use doubleArray
typedef struct VEC Bias;    // the bias vector of lineartransform use doubleArray
typedef struct VEC Derv;   // partial derivatives use doubleArray
typedef struct MAT MOutput; // model-output using double matrix
typedef struct MAT MInput;  // model-input using double matrix
typedef struct MAT Weights; // the weights matrix
typedef struct MAT MDerv;   // the derivatives of each weight
typedef struct MAT Kernel;  // the kernel
typedef struct MTS SInput;
typedef struct MTS SOutput;
typedef struct MTS SKernel;
typedef struct MTS SDerv;
typedef enum Status Sts;    // ok when the functions acts well

Sts ReLU(Input *input, Output *output);
Sts leakyReLU(Input *input, Output *output);
Sts lossCrossEntropy(Label *label, Input *input, Output *output);
Sts ReLU_derivative(Input *input, Derv *derv);
Sts leakyReLU_derivative(Input *input, Derv *derv);
Sts lossCrossEntropy_derivative(Label *label, Input *input, Derv *derv);
Sts softmax(Input *input, Output *output);
Sts softmax_derivative(Input *input, Derv *derv);
Sts sigmoid(Input *input, Derv *derv);
Sts optimizeDoubleVec(Vec *args, Derv *derv, double lr);
Sts optimizeDoubleMat(Mat *args, MDerv *derv, double lr);
Sts noActivation(Input *input, Output *output);
Sts noActivation_derivative(Input *input, Derv *derv);
Sts convolution(MInput *origin, MOutput *dst, Kernel *kernel);
Sts poolingMax(MInput *origin, MOutput *dst, int kernelSize);
Sts flatten(Mts *matrxStack, Vec *dst);

double MSE_single(double label, double output); // loss function for test
double MSE_single_derivative(double label, double output);

#endif
