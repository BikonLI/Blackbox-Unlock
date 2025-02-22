#include "functions.h"

Sts ReLU(Input *input, Output *output)
{
    if (input->length != output->length)
        return ERROR;

    for (size_t i = 0; i < input->length; i++)
    {
        double x = input->array.doubleArray[i];
        output->array.doubleArray[i] = x > 0 ? x : 0;
    }

    return OK;
}

Sts leakyReLU(Input *input, Output *output)
{
    if (input->length != output->length)
        return ERROR;

    for (size_t i = 0; i < input->length; i++)
    {
        double x = input->array.doubleArray[i];
        output->array.doubleArray[i] = x > 0 ? x : .01 * x;
    }

    return OK;
}

Sts lossCrossEntropy(Label *label, Input *input, Output *output)
{
    if ((label->length != input->length) || output->length != 1)
        return ERROR;

    if (label->length == 1)
    {
        int y = label->array.intArray[0];
        double y1 = input->array.doubleArray[0];
        output->array.doubleArray[0] = -y * log(y1) - (1 - y) * log(1 - y1);

        return OK;
    }

    double result = 0;
    for (size_t i = 0; i < label->length; i++)
        result -= label->array.intArray[i] * log(input->array.doubleArray[i]);

    output->array.doubleArray[0] = result;

    return OK;
}

Sts ReLU_derivative(Input *input, Derv *derv)
{
    if (input->length != derv->length)
        return ERROR;

    for (size_t i = 0; i < input->length; i++)
    {
        double x = input->array.doubleArray[i];
        derv->array.doubleArray[i] = x > 0 ? 1 : 0;
    }

    return OK;
}

Sts leakyReLU_derivative(Input *input, Derv *derv)
{
    if (input->length != derv->length)
        return ERROR;

    for (size_t i = 0; i < input->length; i++)
    {
        double x = input->array.doubleArray[i];
        derv->array.doubleArray[i] = x > 0 ? 1 : .01;
    }

    return OK;
}

Sts lossCrossEntropy_derivative(Label *label, Input *input, Derv *derv)
{
    if (label->length != input->length || label->length != derv->length)
        return ERROR;

    if (label->length == 1)
    {
        int y = label->array.intArray[0];
        double y1 = input->array.doubleArray[0];
        derv->array.doubleArray[0] = -(y / (y1 + 1e-8) - (1 - y) / (1 - y1));

        return OK;
    }

    for (size_t i = 0; i < label->length; i++)
    {
        int y = label->array.intArray[i];
        double y1 = input->array.doubleArray[i];
        derv->array.doubleArray[i] = -y / (y1 + 1e8);
    }

    return OK;
}

Sts softmax(Input *input, Output *output)
{
    if (!input || !output || (input->length != output->length))
        return ERROR;

    // travel two times to compute the denominator
    int max = 0;
    for (size_t i = 0; i < input->length; i++)
        max = max >= input->array.doubleArray[i] ? max : input->array.doubleArray[i];

    long long int totalSum = 0;
    for (size_t i = 0; i < input->length; i++)
    {
        long double x = exp(input->array.doubleArray[i] - max);
        totalSum += x;
        output->array.doubleArray[i] = x;
    }

    for (size_t i = 0; i < input->length; i++)
        output->array.doubleArray[i] /= totalSum;

    return OK;
}

Sts sigmoid(Input *input, Derv *derv)
{
    if (!input || !derv || (input->length != derv->length))
        return ERROR;

    for (size_t i = 0; i < input->length; i++)
        derv->array.doubleArray[i] = 1 / (1 + exp(-input->array.doubleArray[i]));

    return OK;
}

Sts optimizeDoubleVec(Vec *args, Derv *derv, double lr)
{
    if (!args || !derv || args->length != derv->length)
        return ERROR;

    for (size_t i = 0; i < args->length; i++)
        args->array.doubleArray[i] -= lr * derv->array.doubleArray[i];

    return OK;
}

Sts optimizeDoubleMat(Mat *args, MDerv *derv, double lr)
{
    Vec vargs, vderv;
    Sts rcode = OK;
    rcode = matTransVet(args, &vargs) || rcode;
    rcode = matTransVet(derv, &vderv) || rcode;

    optimizeDoubleVec(&vargs, &vderv, lr);

    if (rcode == ERROR)
        return ERROR;

    return OK;
}

Sts noActivation(Input *input, Output *output)
{
    if (!input || !output || input->length != output->length)
        return ERROR;

    for (size_t i = 0; i < input->length; i++)
        output->array.doubleArray[i] = input->array.doubleArray[i];

    return OK;
}

Sts noActivation_derivative(Input *input, Derv *derv)
{
    if (!input || !derv || input->length != derv->length)
        return ERROR;

    for (size_t i = 0; i < input->length; i++)
        derv->array.doubleArray[i] = 1;

    return OK;
}

Sts convolution(MInput *origin, MOutput *dst, Kernel *kernel)
{
    int m = origin->row, n = origin->col, m1 = dst->row, n1 = dst->col, k1 = kernel->row, k2 = kernel->col;

    if (k1 != k2 || k1 % 2 == 1 || ((m - k1 + 1) != m1) || ((n - k1 + 1) != n1))
        return ERROR; // if the dimention of the kernel is not fitting the dst

    int start_index = k1 / 2;
    for (int i = start_index; i < m - start_index; i++)
        for (int j = start_index; j < n - start_index; j++)
        {
            int org_row = i - start_index, org_col = j - start_index;
            double cell = 0, total_weight = 0;
            for (int p = 0; p < k1; p++)
                for (int q = 0; q < k1; q++)
                {
                    cell += getDoubleMatrixValue(kernel, p, q) * getDoubleMatrixValue(origin, org_row + p, org_col + q);
                    total_weight += getDoubleMatrixValue(kernel, p, q);
                }

            if (total_weight == 0)
                total_weight += 1e-8;

            setDoubleMatrixValue(dst, i, j, cell / total_weight);
        }

    return OK;
}

Sts poolingMax(MInput *origin, MOutput *dst, int kernelSize)
{
    int m = origin->row, n = origin->col, m1 = dst->row, n1 = dst->col;
    if (!(m % kernelSize == 0 && n % kernelSize == 0 && m / kernelSize == m1 && n / kernelSize == n1))
        return ERROR;

    for (int i = 0, p = 0; i < m1; i++, p += kernelSize) // i, j -> dst; p, q -> origin; s, t -> kernel
    {
        for (int j = 0, q = 0; j < n1; j++, q += kernelSize)
        {
            double currentValue, maxValue = getDoubleMatrixValue(origin, p, q);
            for (int s = 0; s < kernelSize; s++) // find the max in the kernel
            {
                for (int t = 0; t < kernelSize; t++)
                {
                    currentValue = getDoubleMatrixValue(origin, p + s, q + t);
                    maxValue = currentValue > maxValue ? currentValue : maxValue;
                }
            }
            setDoubleMatrixValue(dst, i, j, maxValue);
        }
    }

    return OK;
}

double MSE_single(double label, double output)
{
    return (label - output) * (label - output);
}

double MSE_single_derivative(double label, double output)
{
    return -2 * (label - output);
}
