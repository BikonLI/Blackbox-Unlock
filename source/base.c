#include "base.h"
#include <stdio.h>

char colorMap[][10] = {
    "\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m",
};

Mat *genDoubleMat(int row, int col, double cell)
{
    Mat *matrix = (Mat *)malloc(sizeof(Mat));
    if (!matrix)
        return NULL;

    Sts rcode = initDoubleMat(matrix, row, col, cell);
    if (rcode == ERROR)
    {
        free(matrix);
        return NULL;
    }

    return matrix;
}

Vec *genDoubleVec(int length, double cell)
{
    Vec *vector = (Vec *)malloc(sizeof(Vec));
    if (!vector)
        return NULL;

    Sts rcode = initDoubleVec(vector, length, cell);
    if (rcode == ERROR)
    {
        free(vector);
        return NULL;
    }

    return vector;
}

Mts *genDoubleMts(int channel, int height, int width, double cell)
{
    Mts *matrixStack = (Mts *)malloc(sizeof(Mts));
    if (!matrixStack)
        return NULL;

    Sts rcode = initDoubleMts(matrixStack, channel, height, width, cell);
    if (rcode == ERROR)
    {
        free(matrixStack);
        return NULL;
    }

    return matrixStack;
}

Sts freeMat(Mat *mat)
{
    if (!mat)
        return OK;

    free(mat->array.doubleMatrix);
    free(mat);

    return OK;
}

Sts freeVec(Vec *vec)
{
    if (!vec)
        return OK;

    free(vec->array.doubleArray);
    free(vec);

    return OK;
}

Sts initDoubleMat(Mat *mat, int row, int col, double cell)
{
    if (!mat)
        return ERROR;

    mat->row = row;
    mat->col = col;

    mat->array.doubleMatrix = (double *)malloc(sizeof(double) * row * col);
    if (!mat->array.doubleMatrix)
        return ERROR;

    double bound = sqrt(6.0 / (row + col));

    if (fabs(cell - 0) <= 1e-4)
        for (size_t i = 0; i < row * col; i++)
            mat->array.doubleMatrix[i] = 0;
    else
        for (size_t i = 0; i < row * col; i++)
            mat->array.doubleMatrix[i] = (rand() / (double)RAND_MAX) * 2 * bound - bound;

    return OK;
}

Sts initDoubleVec(Vec *vec, int length, double cell)
{
    if (!vec)
        return ERROR;

    vec->length = length;

    vec->array.doubleArray = (double *)malloc(sizeof(double) * length);
    if (!vec->array.doubleArray)
    {
        free(vec);
        return ERROR;
    }

    for (int i = 0; i < length; i++)
        vec->array.doubleArray[i] = cell;

    return OK;
}

Sts initDoubleMts(Mts *mts, int channel, int height, int width, double cell)
{
    if (!mts)
        return ERROR;

    mts->channel = channel;
    mts->height = height;
    mts->width = width;

    size_t total = channel * height * width;
    mts->array.doubelMatrixStack = (double *)malloc(sizeof(double) * total);
    if (!mts->array.doubelMatrixStack)
    {
        free(mts);
        return ERROR;
    }

    double bound = sqrt(6.0 / (total));

    if (fabs(cell - 0) <= 1e-4)
        for (size_t i = 0; i < total; i++)
            mts->array.doubelMatrixStack[i] = 0;
    else
        for (size_t i = 0; i < total; i++)
            mts->array.doubelMatrixStack[i] = (rand() / (double)RAND_MAX) * 2 * bound - bound;

    return OK;
}

Sts vecTransMat(Vec *vec, Mat *mat, int row, int col)
{
    if (!vec || !mat || row * col != vec->length)
        return ERROR;

    mat->row = row;
    mat->col = col;
    mat->array.doubleMatrix = vec->array.doubleArray;

    return OK;
}

Sts matTransVec(Mat *mat, Vec *vec)
{
    if (!mat || !vec)
        return ERROR;

    vec->length = mat->row * mat->col;
    vec->array.doubleArray = mat->array.doubleMatrix;

    return OK;
}

Sts vecTransMts(Vec *vec, Mts *mts, int channel, int height, int width)
{
    if (!vec || !mts || vec->length != channel * height * width)
        return ERROR;

    mts->channel = channel;
    mts->height = height;
    mts->width = width;
    mts->array.doubelMatrixStack = vec->array.doubleArray;

    return OK;
}

Sts mtsTransVec(Mts *mts, Vec *vec)
{
    if (!mts || !vec)
        return ERROR;

    vec->length = mts->channel * mts->height * mts->width;
    vec->array.doubleArray = mts->array.doubelMatrixStack;

    return OK;
}

Sts mtsSliceMat(Mts *mts, Mat *mat, int channel)
{
    if (!mts || !mat || channel >= mts->channel)
        return ERROR;

    size_t totalEle = mts->height * mts->width;
    size_t start_index = totalEle * channel;

    mat->row = mts->height;
    mat->col = mts->width;
    mat->array.doubleMatrix = &mts->array.doubelMatrixStack[start_index];

    return OK;
}

Sts crossProductDoubleMatrix(Mat *m1, Mat *m2, Mat *result)
{
    if ((m1->col != m2->row) || (result->row != m1->row) || (result->col != m2->col))
        return ERROR;

    int row = m1->row, col = m2->col, mid = m1->col;
    double(*matrix1)[mid] = (double(*)[mid])m1->array.doubleMatrix;
    double(*matrix2)[col] = (double(*)[col])m2->array.doubleMatrix;
    double(*resultMatrix)[col] = (double(*)[col])result->array.doubleMatrix;

    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        {
            double cell = 0;
            for (int k = 0; k < mid; k++)
                cell += matrix1[i][k] * matrix2[k][j];
            resultMatrix[i][j] = cell;
        }

    return OK;
}

Sts addDoubleMatrix(Mat *m1, Mat *m2, Mat *result)
{
    if (!m1 || !m2 || (m1->row != m2->row) || (m1->col != m2->col))
        return ERROR;

    int row = m1->row, col = m2->col, mid = m1->col;
    double(*matrix1)[mid] = (double(*)[mid])m1->array.doubleMatrix;
    double(*matrix2)[col] = (double(*)[col])m2->array.doubleMatrix;
    double(*resultMatrix)[col] = (double(*)[col])result->array.doubleMatrix;

    for (size_t i = 0; i < m1->row; i++)
        for (size_t j = 0; j < m2->col; j++)
            resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];

    return OK;
}

Sts addDoubleVector(Vec *v1, Vec *v2, Vec *result)
{
    if (!v1 || !v2 || v1->length != v2->length)
        return ERROR;

    for (size_t i = 0; i < v1->length; i++)
        result->array.doubleArray[i] = v1->array.doubleArray[i] + v2->array.doubleArray[i];

    return OK;
}

Sts mulDoubleVector(Vec *v1, Vec *v2, Vec *result)
{
    if (!v1 || !v2 || !result || (v1->length != v2->length) || (v1->length != result->length))
        return ERROR;

    for (size_t i = 0; i < v1->length; i++)
        result->array.doubleArray[i] = v1->array.doubleArray[i] * v2->array.doubleArray[i];

    return OK;
}

Sts setDoubleMatrixValue(Mat *m, int row, int col, double cell)
{
    if (!m || row >= m->row || col >= m->col)
        return ERROR;

    double(*matrix)[m->col] = (double(*)[m->col])m->array.doubleMatrix;
    matrix[row][col] = cell;

    return OK;
}

Sts setDoubleMatrixStackValue(Mts *mts, int channel, int height, int width, double cell)
{
    if (!mts || channel >= mts->channel || height >= mts->height || width >= mts->width)
        return ERROR;

    double(*tensor)[mts->height][mts->width] = (double(*)[mts->height][mts->width])mts->array.doubelMatrixStack;
    tensor[channel][height][width] = cell;

    return OK;
}

double getDoubleMatrixValue(Mat *m, int row, int col)
{
    if (!m || row >= m->row || col >= m->col)
        return .0f;

    double(*matrix)[m->col] = (double(*)[m->col])m->array.doubleMatrix;
    return matrix[row][col];
}

double getDoubleMatrixStackValue(Mts *mts, int channel, int height, int width)
{
    if (!mts || channel >= mts->channel || height >= mts->height || width >= mts->width)
        return .0f;

    double(*tensor)[mts->height][mts->width] = (double(*)[mts->height][mts->width])mts->array.doubelMatrixStack;
    return tensor[channel][height][width];
}

Sts printDoubleMatrix(Mat *m)
{
    static int times = 0;
    times++;

    char *color = colorMap[times % 7], *reset = colorMap[0];

    if (!m)
    {
        printf("%s[   0x0%s: null\n%s]%s\n", color, reset, color, reset);
        return OK;
    }

    double(*matrix)[m->col] = (double(*)[m->col])m->array.doubleMatrix;
    printf("%s[   0x%X%s: %d x %d\n", color, m, reset, m->row, m->col);
    for (int i = 0; i < m->row; i++)
    {
        printf("    [");
        for (int j = 0; j < m->col; j++)
            j ? printf(" %.3f", matrix[i][j]) : printf("%.3f", matrix[i][j]);
        printf("],\n");
    }
    printf("%s]%s\n", color, reset);

    return OK;
}

Sts printDoubleVector(Vec *v)
{
    static int times = 0;
    times++;

    char *color = colorMap[times % 7], *reset = colorMap[0];

    if (!v)
    {
        printf("%s[   0x0%s: null   %s]%s\n", color, reset, color, reset);
        return OK;
    }

    printf("%s[   0x%X%s: %d\n", color, v, reset, v->length);
    for (int i = 0; i < v->length; i++)
        i ? printf(" %.3f", v->array.doubleArray[i]) : printf("    %.3f", v->array.doubleArray[i]);
    printf("\n%s]%s\n", color, reset);

    return OK;
}

double doubleaThreshold(double x)
{
    int is_nan = isnan(x), is_inf = isinf(x), is_pos = x > 0, is_out_of_range = fabs(x) > DOUBLE_THRESHOLD;

    if (is_nan) // if it is illegle
        return DOUBLE_DEFAULT;

    if (!(is_nan || is_inf || is_out_of_range)) // if the value is legle and not out of range
        return x;

    if (is_pos) // if it is too big
        return DOUBLE_THRESHOLD;

    if (!is_pos) // if it is too small
        return -DOUBLE_THRESHOLD;

    return x;
}
