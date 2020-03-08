#pragma once
#include <vector>
#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"
#include <algorithm>

#include "Block.h"
using namespace std;


// r_from, spin_dir, tests, dim
static int wallkick_jlstz[4][2][5][2]{
    {
        {{ 0, 0}, {-1, 0}, {-1, 1}, { 0,-2}, {-1,-2}},
        {{ 0, 0}, { 1, 0}, { 1, 1}, { 0,-2}, { 1,-2}},
    },
    {
        {{ 0, 0}, { 1, 0}, { 1,-1}, { 0, 2}, { 1, 2}},
        {{ 0, 0}, { 1, 0}, { 1,-1}, { 0, 2}, { 1, 2}}
    },
    {
        {{ 0, 0}, { 1, 0}, { 1, 1}, { 0,-2}, { 1,-2}},
        {{ 0, 0}, {-1, 0}, {-1, 1}, { 0,-2}, {-1,-2}}
    },
    {
        {{ 0, 0}, {-1, 0}, {-1,-1}, { 0, 2}, {-1, 2}},
        {{ 0, 0}, {-1, 0}, {-1,-1}, { 0, 2}, {-1, 2}}
    }
};

static int wallkick_l[4][2][5][2]{
    {
        {{ 0, 0}, {-2, 0}, { 1, 0}, {-2,-1}, { 1, 2}},
        {{ 0, 0}, {-1, 0}, { 2, 0}, {-1, 2}, { 2,-1}},
    },
    {
        {{ 0, 0}, {-1, 0}, { 2, 0}, {-1, 2}, { 2,-1}},
        {{ 0, 0}, { 2, 0}, {-1, 0}, { 2, 1}, {-1,-2}},
    },
    {
        {{ 0, 0}, { 2, 0}, {-1, 0}, { 2, 1}, {-1,-2}},
        {{ 0, 0}, { 1, 0}, {-2, 0}, { 1,-2}, {-2, 1}},
    },
    {
        {{ 0, 0}, { 1, 0}, {-2, 0}, { 1,-2}, {-2, 1}},
        {{ 0, 0}, {-2, 0}, { 1, 0}, {-2,-1}, { 1, 2}},
    }
};

class Map
{
public:
    Map();
	Map(int _width, int _height);
	Map(PyObject* _data);
	Map(const Map& target);
    Map(int _width, int _height, int* data);
	~Map();

    bool allocate(int _width = 10, int _height = 20);

    bool alloc = true;

    int w, h;

    int* data;
	int at(int x, int y);
	void set(int x, int y, int _data);
	bool fit(int type, Pos position);
	bool put(int type, Pos position);
	Pos rotate(int type, Pos pos, int n);
    Pos drop(int type, Pos pos);
    bool contains(int x, int y);
    PyObject* toArray();

};

