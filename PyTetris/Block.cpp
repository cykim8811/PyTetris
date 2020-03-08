
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include "Block.h"


Block init_rotate_Block(Block* original, int n) { // Return clockwise rotated 'Type'
    int s = original->size;
    Block ret, temp;
    ret.size = original->size;
    temp.size = original->size;

    for (int i = 0; i < s * s; i++) {
        ret.data[i] = original->data[i];
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < s * s; i++) {
            temp.data[i] = ret.data[i];
        }
        for (int x = 0; x < s; x++) {
            for (int y = 0; y < s; y++) {
                ret.data[(s - 1 - y) * s + x] = temp.data[x * s + y];
                // *at(&ret, s - 1 - y, x) = *at(&temp, x, y);
            }
        }
    }
    for (int j = s * s; j < 16; j++) {
        ret.data[j] = false;
    }
    return ret;
}

bool Block::at(int x, int y) {
    return data[x * size + y];
}
