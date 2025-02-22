#include "layers.h"
#include <stdio.h>

Sts initFCL(struct FCL *fcl, size_t neuronNumIn, size_t neuronNumOut, Sts (*activateFunction)(Input *, Output *),
            Sts (*activateFunction_derivative)(Input *, Derv *))
{

    if (!fcl)
        return ERROR;

    fcl->activateFunction = activateFunction;
    fcl->activateFunction_derivative = activateFunction_derivative;
    Sts rcode = OK;

    // init input neurons linearTrans and output neurons
    rcode = initDoubleVec(&fcl->input, neuronNumIn, 0) || rcode;
    rcode = initDoubleVec(&fcl->linearTrans, neuronNumOut, 0) || rcode;
    rcode = initDoubleVec(&fcl->output, neuronNumOut, 0) || rcode;

    // init the derivatives of activate function, i.e. dervOfActivateFunc
    rcode = initDoubleVec(&fcl->dervOfActivateFunc, neuronNumOut, 0) || rcode;

    // init bias and it's derv
    rcode = initDoubleVec(&fcl->bias, neuronNumOut, 0) || rcode;
    rcode = initDoubleVec(&fcl->dervOfBias, neuronNumOut, 0) || rcode;
    rcode = initDoubleVec(&fcl->dervFromLastLayer, neuronNumOut, 0) || rcode;
    rcode = initDoubleVec(&fcl->dervToPreviousLayer, neuronNumIn, 0) || rcode;

    // init weight and it's derv
    rcode = initDoubleMat(&fcl->weight, neuronNumOut, neuronNumIn, 1) || rcode;
    rcode = initDoubleMat(&fcl->dervOfWeight, neuronNumOut, neuronNumIn, 0) || rcode;

    if (rcode == ERROR)
    {
        free(fcl->input.array.doubleArray);
        free(fcl->output.array.doubleArray);
        free(fcl->bias.array.doubleArray);
        free(fcl->dervOfBias.array.doubleArray);
        free(fcl->weight.array.doubleMatrix);
        free(fcl->dervOfWeight.array.doubleMatrix);
        free(fcl->dervFromLastLayer.array.doubleArray);
        free(fcl->linearTrans.array.doubleArray);
        free(fcl->dervToPreviousLayer.array.doubleArray);

        return ERROR;
    }

    return OK;
}

Sts forwardFCL(struct FCL *fcl)
{
    if (!fcl)
        return ERROR;

    size_t numIn = fcl->input.length, numOut = fcl->output.length;

    Sts rcode = OK;
    // bind input with m1, linearTrans with m2 for computes
    rcode = vecTransMat(&fcl->input, &fcl->m1, numIn, 1) || rcode;
    rcode = vecTransMat(&fcl->linearTrans, &fcl->m2, numOut, 1) || rcode;

    // y = Wx + b
    rcode = crossProductDoubleMatrix(&fcl->weight, &fcl->m1, &fcl->m2) || rcode;
    rcode = addDoubleVector(&fcl->linearTrans, &fcl->bias, &fcl->linearTrans) || rcode;

    // output = act(y)
    rcode = fcl->activateFunction(&fcl->linearTrans, &fcl->output) || rcode;

    if (rcode == ERROR)
        return ERROR;

    return OK;
}

Sts backwardFCL(struct FCL *fcl, double lr)
{
    if (!fcl)
        return ERROR;

    size_t numIn = fcl->input.length, numOut = fcl->output.length;

    Sts rcode = OK;
    // get the derivatives of activate function
    // multiple the derv from last layer and the derv of activate function and save in the dervOfBias
    rcode = fcl->activateFunction_derivative(&fcl->linearTrans, &fcl->dervOfActivateFunc) || rcode;
    rcode = mulDoubleVector(&fcl->dervFromLastLayer, &fcl->dervOfActivateFunc, &fcl->dervOfBias) || rcode;

    // dervOfActivateFun put in to m1 and x^T put it in to m2 (dervOfWeight = dervOfActivateFun x X^T)
    rcode = vecTransMat(&fcl->dervOfBias, &fcl->m1, numOut, 1) || rcode;
    rcode = vecTransMat(&fcl->input, &fcl->m2, 1, numIn) || rcode;
    rcode = crossProductDoubleMatrix(&fcl->m1, &fcl->m2, &fcl->dervOfWeight) || rcode;

    // dervOfBias put into m1 and use dervOfBias left cross product weight
    // put dervToPreviousLayer to m2 to be the result
    rcode = vecTransMat(&fcl->dervOfBias, &fcl->m1, 1, numOut) || rcode;
    rcode = vecTransMat(&fcl->dervToPreviousLayer, &fcl->m2, 1, numIn) || rcode;
    rcode = crossProductDoubleMatrix(&fcl->m1, &fcl->weight, &fcl->m2) || rcode;

    // start optimizing weight matrix and bias vector
    rcode = optimizeDoubleVec(&fcl->bias, &fcl->dervOfBias, lr) || rcode;
    rcode = optimizeDoubleMat(&fcl->weight, &fcl->dervOfWeight, lr) || rcode;

    if (rcode == ERROR)
        return ERROR;

    return OK;
}

Sts initCVL(struct CVL *cvl, size_t channelIn, size_t rowIn, size_t colIn, size_t kernelSize)
{
    if (!cvl)
        return ERROR;

    size_t rowOut = rowIn - kernelSize, colOut = colIn - kernelSize;
    Sts rcode = OK;
    rcode = initDoubleMts(&cvl->inputs, channelIn, rowIn, colIn, 0) || rcode;
    rcode = initDoubleMts(&cvl->outputs, channelIn, rowOut, colOut, 0) || rcode;
    rcode = initDoubleMts(&cvl->kernels, channelIn, kernelSize, kernelSize, 1) || rcode;
    rcode = initDoubleMts(&cvl->dervsFromLastLayer, channelIn, rowOut, colOut, 0) || rcode;
    rcode = initDoubleMts(&cvl->dervsToPreviousLayer, channelIn, rowIn, colIn, 0) || rcode;
    rcode = initDoubleMts(&cvl->dervsOfKernels, channelIn, kernelSize, kernelSize, 0) || rcode;

    if (rcode == ERROR)
    {
        free(&cvl->inputs.array.doubelMatrixStack);
        free(&cvl->outputs.array.doubelMatrixStack);
        free(&cvl->kernels.array.doubelMatrixStack);
        free(&cvl->dervsFromLastLayer.array.doubelMatrixStack);
        free(&cvl->dervsToPreviousLayer.array.doubelMatrixStack);
        free(&cvl->dervsOfKernels.array.doubelMatrixStack);

        return ERROR;
    }

    return OK;
}

Sts forwardCVL(struct CVL *cvl)
{
    if (!cvl)
        return ERROR;

    Mts *inputs = &cvl->inputs, *outputs = &cvl->outputs, *kernels = &cvl->kernels;

    Sts rcode = OK;
    for (int i = 0; i < inputs->channel; i++)
    {
        rcode = mtsSliceMat(inputs, &cvl->m1, i) || rcode;
        rcode = mtsSliceMat(outputs, &cvl->m2, i) || rcode;
        rcode = mtsSliceMat(kernels, &cvl->m3, i) || rcode;
        rcode = convolution(&cvl->m1, &cvl->m2, &cvl->m3) || rcode;
    }

    if (rcode == ERROR)
        return ERROR;

    return OK;
}

Sts backwardCVL(struct CVL *cvl, double lr)
{
    if (!cvl)
        return ERROR;

    Mts *dervsFromLastLayers = &cvl->dervsFromLastLayer, *dervsOfKernels = &cvl->dervsOfKernels;

    for (int i = 0; i < dervsOfKernels->channel; i++)
    {
        mtsSliceMat(dervsFromLastLayers, &cvl->m1, i);
        
    }
     




}
