/**
 * @file demo4.c
 * @author luwangguerde@163.com
 * @brief To fit vectors
 * @version 0.1
 * @date 2024-11-20
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cnn.h"
#include <stdio.h>
#include <windows.h>

/**
 * Now we are going to fit a vector function, we need to put a vector into the model
 * and get a new vector from the model output. Actually the steps are as same as fitting a
 * unary function. Only thing that we need to consider is the loss function and it's derivative.
 * Here we still use the square of the difference to be the lossValue.
 *
 */

void fitFunc_demo4(Vec *input, Vec *output);
double loss(Vec *output, Vec *real);
void loss_derivative(Vec *output, Vec *real, Vec *derv);

int main_demo4(int argc, char const *argv[])
{
    struct FCL fcl1, fcl2;
    Vec *input, *output, real;

    initFCL(&fcl1, 2, 50, leakyReLU, leakyReLU_derivative);
    initFCL(&fcl2, 50, 2, noActivation, noActivation_derivative);
    initDoubleVec(&real, 2, 0);

    fcl2.input.array.doubleArray = fcl1.output.array.doubleArray;
    fcl1.dervFromLastLayer.array.doubleArray = fcl2.dervToPreviousLayer.array.doubleArray;

    input = &fcl1.input;
    output = &fcl2.output;

    double lossValue = 1;
    int loss_hit_times = 0;
    int time = 100;
    while (loss_hit_times <= 20)
    {
        input->array.doubleArray[0] = (2.0 * (double)rand() / RAND_MAX - 1.0) * .1; // [-.1, .1]
        input->array.doubleArray[1] = (2.0 * (double)rand() / RAND_MAX - 1.0) * .1;
        forwardFCL(&fcl1);
        forwardFCL(&fcl2);

        fitFunc_demo4(input, &real);
        lossValue = loss(output, &real);
        loss_derivative(output, &real, &fcl2.dervFromLastLayer);

        printf("\033[H\033[J"); // clear screen
        printf("input: \n");
        printDoubleVector(input);
        printf("\noutput: \n");
        printDoubleVector(output);
        printf("\nreal: \n");
        printDoubleVector(&real);
        printf("\n");
        printf("lossValue = %f\n", lossValue);

        if (time > 0) // observe the first 100 times
        {
            time--;
            Sleep(100);
        }

        backwardFCL(&fcl2, .01);
        backwardFCL(&fcl1, .01);

        loss_hit_times += lossValue < 1e-3 ? 1 : -loss_hit_times;
    }

    system("pause");
    return 0;
}

void fitFunc_demo4(Vec *input, Vec *output)
{
    if (input->length != 2 || output->length != 2)
        return;

    output->array.doubleArray[0] = 2 * input->array.doubleArray[1];
    output->array.doubleArray[1] = .5 * input->array.doubleArray[0] + 1;
}

double loss(Vec *output, Vec *real)
{
    if (output->length != 2 || real->length != 2)
        return -1;

    double d1 = output->array.doubleArray[0] - real->array.doubleArray[0];
    double d2 = output->array.doubleArray[1] - real->array.doubleArray[1];

    return (d1 * d1 + d2 * d2) / 2;
}

void loss_derivative(Vec *output, Vec *real, Vec *derv)
{
    if (output->length != 2 || real->length != 2 || derv->length != 2)
        return;

    derv->array.doubleArray[0] = output->array.doubleArray[0] - real->array.doubleArray[0];
    derv->array.doubleArray[1] = output->array.doubleArray[1] - real->array.doubleArray[1];
}
