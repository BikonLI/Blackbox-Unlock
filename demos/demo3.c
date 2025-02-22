/**
 * @file demo2.c
 * @author luwangguerde@163.com
 * @brief A demo using fully connected layer and activation function to fit (y = x^2)
 * @version 0.1
 * @date 2024-11-12
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cnn.h"
#include <stdio.h>
#include <windows.h>

#define HIDEN_NEUROS_1 100 // neuron numbers of first layer output
#define HIDEN_NEUROS_2 100 // neuron numbers of last layer input
#define LOSS_MAX 1e-3      // value where loss will convergence
#define TRAIN_RANGE .5     // the train dataset range
#define LR .05             // learning rate

/**
 * To fit a function like y = x ^ 2 + x + 1 seems to be more difficult, because it
 * adds non-linear relationships. To solve this, we need to add a function
 * called activate function, to provide the no-linear relation. Why don't you try to
 * modify different values above to see the differeces?
 * Observe the convergence time cost and the where the lossValue convergence
 */

float fitFunc_demo3(float x); // the func that we attempt to fit, here is y = x * x + x + 1

int main_demo3(int argc, char const *argv[])
{
    struct FCL fcl1, fcl2, fcl3;

    initFCL(&fcl1, 1, HIDEN_NEUROS_1, leakyReLU, leakyReLU_derivative);
    initFCL(&fcl2, HIDEN_NEUROS_1, HIDEN_NEUROS_2, leakyReLU, leakyReLU_derivative);
    initFCL(&fcl3, HIDEN_NEUROS_2, 1, noActivation, noActivation_derivative);

    fcl2.input.array.doubleArray = fcl1.output.array.doubleArray; // connect two layers
    fcl3.input.array.doubleArray = fcl2.output.array.doubleArray;
    fcl2.dervFromLastLayer.array.doubleArray = fcl3.dervToPreviousLayer.array.doubleArray;
    fcl1.dervFromLastLayer.array.doubleArray = fcl2.dervToPreviousLayer.array.doubleArray;

    double lossValue = 1;
    int loss_hit_times = 0;
    while (loss_hit_times <= 20)
    {
        double input = (2.0 * (double)rand() / RAND_MAX - 1.0) * TRAIN_RANGE; // generate numbers between [-5, 5]
        double output, real;

        fcl1.input.array.doubleArray[0] = input;
        forwardFCL(&fcl1);
        forwardFCL(&fcl2);
        forwardFCL(&fcl3);
        output = fcl3.output.array.doubleArray[0];
        real = fitFunc_demo3(input);
        lossValue = MSE_single(real, output);

        printf("\033[H\033[J"); // clear screen
        printf("input=%.5f, output=%.5f, real=%.5f\n", input, output, real);
        printf("lossValue: %.5f\n", lossValue);
        Sleep(10);

        fcl3.dervFromLastLayer.array.doubleArray[0] = doubleaThreshold(MSE_single_derivative(real, output));
        backwardFCL(&fcl3, LR);
        backwardFCL(&fcl2, LR);
        backwardFCL(&fcl1, LR);

        loss_hit_times += lossValue < LOSS_MAX ? 1 : -loss_hit_times;
    }

    system("pause");
    return 0;
}

float fitFunc_demo3(float x)
{
    return x * x + x + 1;
}
