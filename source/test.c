#include "cnn.h"
#include <stdio.h>
int main(int argc, char const *argv[])
{
    Mts *mts = genDoubleMts(3, 3, 3, 0);
    // for (int i = 0; i < 3; i++)
    // {
    //     print
    // }
 
    int l = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++, l++)
                setDoubleMatrixStackValue(mts, i, j, k, l);

    Vec vec = {
        .length = 28,
        .array.doubleArray = mts->array.doubelMatrixStack,
    };

    printDoubleVector(&vec);

    system("pause");
    return 0;
}

