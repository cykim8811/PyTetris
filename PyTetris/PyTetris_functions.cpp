
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include "PyTetris_functions.h"

PyObject* PyTetris_available_spots(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* map = NULL;
    int type;

    static char* keywords[] = { "map", "type", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keywords, &map, &type)) {
        return NULL;
    }
    vector<Pos> res = available_spots(Map((PyObject*)map), type);


    int nd = 2;
    npy_intp dim[2] = { res.size(), 3 };
    PyArrayObject* ret = (PyArrayObject*)PyArray_ZEROS(nd, &dim[0], NPY_INT, 0);

    for (int i = 0; i < res.size(); i++) {
        *(((int*)ret->data) + i * 3 + 0) = res[i].x;
        *(((int*)ret->data) + i * 3 + 1) = res[i].y;
        *(((int*)ret->data) + i * 3 + 2) = res[i].r;
    }

    return (PyObject*)ret;
}


PyObject* PyTetris_available_spots_strict(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* map = NULL;
    int type;
    PyObject* from_node = NULL;

    static char* keywords[] = { "map", "type", "from", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiO", keywords, &map, &type, &from_node)) {
        return NULL;
    }
    int f_x, f_y, f_r;
    if (!PyArg_ParseTuple(from_node, "iii", &f_x, &f_y, &f_r)) {
        return NULL;
    }

    vector<Pos> res = available_spots_strict(Map((PyObject*)map), type, Pos{ f_x, f_y, f_r });

    Py_DECREF(from_node);

    int nd = 2;
    npy_intp dim[2] = { res.size(), 3 };
    PyArrayObject* ret = (PyArrayObject*)PyArray_ZEROS(nd, &dim[0], NPY_INT, 0);

    for (int i = 0; i < res.size(); i++) {
        *(((int*)ret->data) + i * 3 + 0) = res[i].x;
        *(((int*)ret->data) + i * 3 + 1) = res[i].y;
        *(((int*)ret->data) + i * 3 + 2) = res[i].r;
    }

    return (PyObject*)ret;
}

PyObject* get_block(PyObject* self, PyObject* args, PyObject* kwargs) {
    int type, angle;

    static char* keywords[] = { "type", "angle", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", keywords, &type, &angle)) {
        return NULL;
    }

    if (type < 0 || type >= 7 || angle < 0 || angle >= 4) {
        return NULL;
    }

    Block* target = &Tile[type][angle];

    int nd = 2;
    npy_intp dim[2] = { target->size, target->size };
    PyArrayObject* ret = (PyArrayObject*)PyArray_ZEROS(nd, &dim[0], NPY_INT, 0);

    for (int i = 0; i < target->size * target->size; i++) {
        *(((int*)ret->data) + i) = target->data[i];
    }

    return (PyObject*)ret;

}


PyObject* PyTetris_search_path_op(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* map = NULL;
    int type;
    PyObject* from_object = NULL;
    PyObject* to_object = NULL;

    static char* keywords[] = { "map", "type", "from", "to", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiOO", keywords, &map, &type, &from_object, &to_object )) {
        return NULL;
    }
    int f_x, f_y, f_r;
    if (!PyArg_ParseTuple(from_object, "iii", &f_x, &f_y, &f_r)) {
        Py_DECREF(from_object);
        return NULL;
    }
    Py_DECREF(from_object);
    int t_x, t_y, t_r;
    if (!PyArg_ParseTuple(to_object, "iii", &t_x, &t_y, &t_r)) {
        Py_DECREF(to_object);
        return NULL;
    }
    Py_DECREF(to_object);


    path_op res = search_path_and_op(Map((PyObject*)map), type,
        Pos{ f_x, f_y, f_r }, Pos{ t_x, t_y, t_r });


    int nd = 2;
    npy_intp dim[2] = { res.path.size(), 3 };
    PyArrayObject* retp = (PyArrayObject*)PyArray_ZEROS(nd, &dim[0], NPY_INT, 0);

    for (int i = 0; i < res.path.size(); i++) {
        *(((int*)retp->data) + i * 3 + 0) = res.path[i].x;
        *(((int*)retp->data) + i * 3 + 1) = res.path[i].y;
        *(((int*)retp->data) + i * 3 + 2) = res.path[i].r;
    }

    nd = 1;
    dim[0] = res.operations.size();
    PyArrayObject* reto = (PyArrayObject*)PyArray_ZEROS(nd, &dim[0], NPY_INT, 0);

    for (int i = 0; i < res.operations.size(); i++) {
        *(((int*)reto->data) + i) = res.operations[i];
    }

    PyObject* ret = Py_BuildValue("OO", retp, reto);
    Py_DECREF(retp);
    Py_DECREF(reto);
    return ret;
}