#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include "Map.h"

Map::Map() {
    alloc = false;
}

Map::Map(int _width, int _height) {
    data.resize(_width * _height);
    w = _width;
    h = _height;
    alloc = true;
}

Map::Map(int _width, int _height, int* _data) {
    data.resize(w * h);
    for (int i = 0; i < w * h; i++) {
        data[i] = *(_data + i);
    }
    w = _width;
    h = _height;
}

Map::Map(int _width, int _height, vector<int> _data) {
    data = _data;
    w = _width;
    h = _height;
    alloc = true;
}

Map::Map(PyObject* _data) {
    PyArrayObject* target = (PyArrayObject*)_data;
    w = (int)target->dimensions[0];
    h = (int)target->dimensions[1];
    data.resize(w * h);
    for (int i = 0; i < w * h; i++) {
        data[i] = *((int*)target->data + i);
    }
    alloc = true;
}

Map::Map(const Map& target) {
    w = target.w;
    h = target.h;
    data = target.data;
    alloc = true;
}

Map::~Map() {
    alloc = false;
}

bool Map::allocate(int _width, int _height) {
    if (alloc) {
        return false;
    }
    alloc = true;
    w = _width;
    h = _height;
    data.resize(w * h);
    return true;
}

int Map::at(int x, int y) {
	return data[x * h + y];
}

void Map::set(int x, int y, int _data) {
    data[x * h + y] = _data;
}

bool Map::fit(int type, Pos position) {
    Block* block_data = &Tile[type][position.r];

    for (int dx = 0; dx < block_data->size; dx++) {
        for (int dy = 0; dy < block_data->size; dy++) {
            if (!block_data->at(dx, dy)) continue;
            if (position.x + dx < 0 || position.x + dx >= w || position.y + dy >= h) return false;
            if (position.y + dy < 0) continue;
            if (at(position.x + dx, position.y + dy)) return false;
        }
    }

    return true;
}

bool Map::put(int type, Pos position) {
    Block tile = Tile[type][position.r];

    for (int x = 0; x < tile.size; x++) {
        for (int y = 0; y < tile.size; y++) {
            const int tx = x + position.x,
                ty = y + position.y;
            if (!tile.at(x, y)) continue;
            if (tx < 0 || tx >= w || ty < 0 || ty >= h)
                return false;
            if (at(tx, ty) != 0)
                return false;
            set(tx, ty, type + 1);
        }
    }
    return true;
}

Pos Map::rotate(int type, Pos pos, int n) {
    int rp = pos.r + n;

    while (rp < 0) rp += 4;
    rp = rp % 4;
    
    int dirind;
    if (rp == 1)
        dirind = 0;
    else
        dirind = 1;

    if (type == 3)
        return Pos{ pos.x, pos.y, rp };
    else if (type == 0) {
        for (int i = 0; i < 5; i++) {
            Pos cp{ pos.x + wallkick_l[pos.r][dirind][i][0], pos.y - wallkick_l[pos.r][dirind][i][1], rp % 4 };
            if (fit(type, cp)) {
                return cp;
            }
        }
    }
    else {
        for (int i = 0; i < 5; i++) {
            Pos cp{ pos.x + wallkick_jlstz[pos.r][dirind][i][0], pos.y - wallkick_jlstz[pos.r][dirind][i][1], rp % 4 };
            if (fit(type, cp)) {
                return cp;
            }
        }
    }
    return Pos{ 0, 0, -1 };
}

Pos Map::drop(int type, Pos pos) {
    Pos newpos = { pos.x, pos.y + 1, pos.r };
    if (!fit(type, newpos))
        return Pos{ 0, 0, -1 };
    while (fit(type, newpos)) {
        pos = newpos;
        newpos = { pos.x, pos.y + 1, pos.r };
    }
    return pos;
}

bool Map::contains(int x, int y) {
    return (x >= 0 && x < w &&
        y >= 0 && y < h);
}

PyObject* Map::toArray() {

    int nd = 2;
    npy_intp dim[2] = { w, h };
    PyArrayObject* ret = (PyArrayObject*)PyArray_ZEROS(nd, &dim[0], NPY_INT, 0);

    for (int i = 0; i < w * h; i++) {
        *(((int*)ret->data) + i) = data[i];
    }
    return (PyObject*)ret;
}