/**
 * @file matrix.h
 * @author luwangguerde@163.com
 * @brief Matrix operations
 * @version 0.1
 * @date 2024-11-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <stdlib.h>

#define DOUBLE_THRESHOLD 0xf
#define DOUBLE_DEFAULT 0x0

struct VEC // vector with lenth dimention
{
    union {
        int *intArray;
        char *charArray;
        float *floatArray;
        double *doubleArray;
    } array;

    size_t length;
};

struct MAT // matrix with shape in row col
{
    union {
        int *intArray;
        char *charArray;
        float *floatArray;
        double *doubleMatrix;
    } array;

    size_t row;
    size_t col;
};

struct MTS // matrix stack
{
    union {
        int *intArray;
        char *charArray;
        float *floatArray;
        double *doubelMatrixStack;
    } array;

    size_t channel;
    size_t height;
    size_t width;
};

enum Status
{
    OK,
    ERROR
};

typedef struct MAT Mat;
typedef struct VEC Vec;
typedef struct MTS Mts;
typedef enum Status Sts;

Mat *genDoubleMat(int row, int col, double cell); // create and init
Vec *genDoubleVec(int length, double cell);
Mts *genDoubleMts(int channel, int height, int width, double cell);
Sts freeMat(Mat *mat);
Sts freeVec(Vec *vec);
Sts initDoubleMat(Mat *mat, int row, int col, double cell); // only init the array
Sts initDoubleVec(Vec *vec, int length, double cell);
Sts initDoubleMts(Mts *mts, int channel, int height, int width, double cell);
Sts vecTransMat(Vec *vec, Mat *mat, int row, int col); // trans function will not copy data
Sts matTransVec(Mat *mat, Vec *vec);
Sts vecTransMts(Vec *vec, Mts *mts, int channel, int height, int width);
Sts mtsTransVec(Mts *mts, Vec *vec);
Sts mtsSliceMat(Mts *mts, Mat *mat, int channel);
Sts crossProductDoubleMatrix(Mat *m1, Mat *m2, Mat *result); // the result shouldn't be one of m1 or m2
Sts addDoubleMatrix(Mat *m1, Mat *m2, Mat *result);          // the result could be one of m1 or m2
Sts addDoubleVector(Vec *v1, Vec *v2, Vec *result);
Sts mulDoubleVector(Vec *v1, Vec *v2, Vec *result);
Sts setDoubleMatrixValue(Mat *m, int row, int col, double cell);
Sts setDoubleMatrixStackValue(Mts *mts, int channel, int height, int width, double cell);
double getDoubleMatrixValue(Mat *m, int row, int col);
double getDoubleMatrixStackValue(Mts *mts, int channel, int height, int width);
Sts printDoubleMatrix(Mat *m);
Sts printDoubleVector(Vec *v);
double doubleaThreshold(double x); // examine whether the number is inf or nan

#endif