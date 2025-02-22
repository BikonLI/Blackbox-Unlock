#ifndef LAYERS_H
#define LAYERS_H

#include "base.h"
#include "functions.h"

struct FCL // fully connected layer, taking charge of three operations (cross weights, add bias, activate)
{
    Input input;               // input vector
    Output linearTrans;        // the middle layer after linear trans, keeping the value of (Wx + b)
    Output output;             // output vector, σ(Wx + b)
    Weights weight;            // the weights matrix
    Bias bias;                 // the bias vector
    Derv dervOfBias;          // the derivatives of the bias vector
    MDerv dervOfWeight;        // the derivatives of the weight matrix
    Derv dervFromLastLayer;   // the derivatives from last layer, normally from ouput layer
    Derv dervToPreviousLayer; // the derivatives to previous layer during the backward
    Derv dervOfActivateFunc;  // the derivatives of the activation function

    /*
    This two matrix is only for matrix computes,
    it sholdn't be operated manually (with out using functions).
    */
    Mat m1; // broadcast a vector to a matrix to execute matrix operations
    Mat m2;
    Sts (*activateFunction)(Input *, Output *);           // the pointer of the activate function
    Sts (*activateFunction_derivative)(Input *, Derv *); // the pointer of the derivative function
};

struct CVL // convolutional layer
{
    SInput inputs;
    SOutput outputs;
    SKernel kernels;

    SDerv dervsOfKernels;
    SDerv dervsFromLastLayer;
    SDerv dervsToPreviousLayer;

    Mat m1;
    Mat m2;
    Mat m3;
};

struct OL // output layer
{
    Input input;
    Output output;
};

// all struct does not provide create operations
Sts initFCL(struct FCL *fcl, size_t neuronNumIn, size_t neuronNumOut, Sts (*activateFunction)(Input *, Output *),
            Sts (*activateFunction_derivative)(Input *, Derv *));
Sts forwardFCL(struct FCL *fcl);
Sts backwardFCL(struct FCL *fcl, double lr);

Sts initCVL(struct CVL *cvl, size_t channelIn, size_t rowIn, size_t colIn, size_t kernelSize);
Sts forwardCVL(struct CVL *cvl);
Sts backwardCVL(struct CVL *cvl, double lr);

#endif
