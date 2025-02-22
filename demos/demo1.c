/**
 * @file demo1.c
 * @author luwangguerde@163.com
 * @brief A demo shows that how to use a FCL to fit (y = 3x + 1)
 * @version 0.1
 * @date 2024-11-11
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cnn.h"
#include <stdio.h>
#include <windows.h> // to best demonstrate, using `Sleep` to slow down convergence

/**
 * To fit a line is really simple, only one input and one output can deal with it.
 * We use MSE_single to compute the loss.
 */

float fitFunc_demo1(float x);                     // the func that we attempt to fit, here is y = 3x + 1

int main_demo1(int argc, char const *argv[])
{
    struct FCL fcl;
    double lossValue = 1; // init MSE_singleValue as 1

    // Because y = x is a linear function, so it is not needed to add a activate function.
    initFCL(&fcl, 1, 1, noActivation, noActivation_derivative);

    while (lossValue > 1e-5)
    {
        double input = (2.0 * (double)rand() / RAND_MAX - 1.0) * 5; // generate numbers between [-5, 5]
        double output, real;

        fcl.input.array.doubleArray[0] = input;   // put the number into the model
        forwardFCL(&fcl);                         // forward
        output = fcl.output.array.doubleArray[0]; // get the output from the model
        real = fitFunc_demo1(input);                    // get the label(real) according to the fitFunc_demo1
        lossValue = MSE_single(real, output);           // compute MSE_single

        printf("\tnow\ttarget\t\n");
        printf("weight\t%.4f\t%d \nbias\t%.4f\t%d \nloss\t%.4f\t%d", fcl.weight.array.doubleMatrix[0], 3,
               fcl.bias.array.doubleArray[0], 1, lossValue, 0);
        Sleep(100);
        printf("\033[H\033[J"); // clear screen

        fcl.dervFromLastLayer.array.doubleArray[0] = MSE_single_derivative(real, output); // put the MSE_single in to the layer
        backwardFCL(&fcl, .01);                                               // optimize
    }

    printf("fitting finished!\n");
    printf("weight matrix:\n");
    printDoubleMatrix(&fcl.weight);
    printf("bias vector:\n");
    printDoubleVector(&fcl.bias);
    printf("loss=%f\n", lossValue);

    system("pause");

    return 0;
}

float fitFunc_demo1(float x)
{
    return 3 * x + 1;
}
