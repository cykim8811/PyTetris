#pragma once

#include <vector>
#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

typedef struct Pos {
    int x, y, r;
};

typedef struct Block {
    int size;
    bool data[16];
    bool at(int x, int y);
};

Block init_rotate_Block(Block* original, int n);

static Block Temp[7]{
    Block{4,
        {
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        }
    },
    Block{3,
        {
        1, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 0, 0, 0, 0, 0, 0
        }
    },
    Block{3,
        {
        0, 1, 0,
        0, 1, 0,
        1, 1, 0,
        0, 0, 0, 0, 0, 0, 0
        }
    },
    Block{2,
        {
        1, 1,
        1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        }
    },
    Block{3,
        {
        0, 1, 0,
        1, 1, 0,
        1, 0, 0,
        0, 0, 0, 0, 0, 0, 0
        }
    },
    Block{3,
        {
        0, 1, 0,
        1, 1, 0,
        0, 1, 0,
        0, 0, 0, 0, 0, 0, 0
        }
    },
    Block{3,
        {
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0, 0, 0, 0, 0, 0, 0
        }
    },
};
static Block Tile[7][4]{
    {init_rotate_Block(&Temp[0], 0), init_rotate_Block(&Temp[0], 1), init_rotate_Block(&Temp[0], 2), init_rotate_Block(&Temp[0], 3)},
    {init_rotate_Block(&Temp[1], 0), init_rotate_Block(&Temp[1], 1), init_rotate_Block(&Temp[1], 2), init_rotate_Block(&Temp[1], 3)},
    {init_rotate_Block(&Temp[2], 0), init_rotate_Block(&Temp[2], 1), init_rotate_Block(&Temp[2], 2), init_rotate_Block(&Temp[2], 3)},
    {init_rotate_Block(&Temp[3], 0), init_rotate_Block(&Temp[3], 1), init_rotate_Block(&Temp[3], 2), init_rotate_Block(&Temp[3], 3)},
    {init_rotate_Block(&Temp[4], 0), init_rotate_Block(&Temp[4], 1), init_rotate_Block(&Temp[4], 2), init_rotate_Block(&Temp[4], 3)},
    {init_rotate_Block(&Temp[5], 0), init_rotate_Block(&Temp[5], 1), init_rotate_Block(&Temp[5], 2), init_rotate_Block(&Temp[5], 3)},
    {init_rotate_Block(&Temp[6], 0), init_rotate_Block(&Temp[6], 1), init_rotate_Block(&Temp[6], 2), init_rotate_Block(&Temp[6], 3)}
};

static int TileColor[7][3]{
    {48, 210, 230},
    {32, 48, 200},
    {230, 180, 48},
    {240, 240, 48},
    {32, 170, 72},
    {210, 64, 235},
    {200, 48, 42},
};