/**
 * @file demo2.c
 * @author luwangguerde@163.com
 * @brief To fit (y = x) with two layers
 * @version 0.1
 * @date 2024-11-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "cnn.h"
#include <stdio.h>
#include <windows.h>

double fitFunc_demo2(double x);

int main_demo2(int argc, char const *argv[])
{
    struct FCL fcl1, fcl2;
    double input, output, real;
    input = (2.0 * (double)rand() / RAND_MAX - 1.0) * 10;

    initFCL(&fcl1, 1, 2, noActivation, noActivation_derivative);
    initFCL(&fcl2, 2, 1, noActivation, noActivation_derivative);

    fcl2.input.array.doubleArray = fcl1.output.array.doubleArray;
    fcl1.dervFromLastLayer.array.doubleArray = fcl2.dervToPreviousLayer.array.doubleArray;

    int loss_hit_times = 0;
    double lossValue = 1;
    while (loss_hit_times <= 20) // breaking condition comes to loss convergence
    {
        input = (2.0 * (double)rand() / RAND_MAX - 1.0) * 10;
        input = 1;

        printf("\033[H\033[J"); // clear screen

        lossValue = MSE_single(fitFunc_demo2(input), output);
        fcl1.input.array.doubleArray[0] = input;
        real = fitFunc_demo2(input);
        forwardFCL(&fcl1);
        forwardFCL(&fcl2);
        output = fcl2.output.array.doubleArray[0];

        printf("input = %f\n", input);
        printf("output = %f\n", output);
        printf("lossValue=%f\n", lossValue);
        Sleep(100);

        fcl2.dervFromLastLayer.array.doubleArray[0] = // examine if is legal
            doubleaThreshold(MSE_single_derivative(fitFunc_demo2(input), output));
        backwardFCL(&fcl2, .01);
        backwardFCL(&fcl1, .01);

        loss_hit_times += lossValue < 1e-5;
    }

    printDoubleMatrix(&fcl1.weight);
    printDoubleVector(&fcl1.bias);
    printDoubleMatrix(&fcl2.weight);
    printDoubleVector(&fcl2.bias);

    system("pause");
    return 0;
}

double fitFunc_demo2(double x)
{
    return x;
}
