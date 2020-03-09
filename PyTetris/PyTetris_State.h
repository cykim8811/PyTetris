#pragma once

#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

#include "Player.h"
#include "Analyzer.h"

static int combo_score[12] = {
    0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4
};

static int clear_score[5] = {
    0, 0, 1, 2, 4
};

typedef struct PyState :PyObject {
    PyObject_HEAD;
    PyState() {}
    Map screen;
    vector<int> block_next;
    int block_hold;
    bool hold_used;

    int combo;
    bool btb;

    vector<int> bag;
};

void PyState_dealloc(PyState* self);
PyObject* PyState_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int PyState_init(PyState* self, PyObject* args, PyObject* kwds);


static PyMemberDef PyState_members[] = {
    {"block_hold", T_INT, offsetof(PyState, block_hold), 0, ""},
    {"hold_used", T_BOOL, offsetof(PyState, hold_used), 0, ""},
    {"combo", T_INT, offsetof(PyState, combo), 0, ""},
    {"btb", T_BOOL, offsetof(PyState, btb), 0, ""},
    {NULL}  /* Sentinel */
};

PyObject* PyState_get_block_next(PyState* self, PyObject* args);
PyObject* PyState_set_block_next(PyState* self, PyObject* args);
PyObject* PyState_get_screen(PyState* self, PyObject* Py_UNUSED(ignore));
PyObject* PyState_set_screen(PyState* self, PyObject* args);
PyObject* PyState_get_bag(PyState* self, PyObject* Py_UNUSED(ignore));
PyObject* PyState_set_bag(PyState* self, PyObject* args);

PyObject* PyState_copy(PyState* self, PyObject* Py_UNUSED(ignore));

PyObject* PyState_transitions(PyState* self, PyObject* args);
PyObject* PyState_hold(PyState* self, PyObject* Py_UNUSED(ignore));


static PyMethodDef PyState_methods[] = {
    {"get_block_next", (PyCFunction)PyState_get_block_next, METH_VARARGS, "" },
    {"get_screen", (PyCFunction)PyState_get_screen, METH_NOARGS, "" },
    {"set_block_next", (PyCFunction)PyState_set_block_next, METH_VARARGS, "" },
    {"set_screen", (PyCFunction)PyState_set_screen, METH_VARARGS, "" },
    {"set_bag", (PyCFunction)PyState_set_bag, METH_VARARGS, "" },
    {"get_bag", (PyCFunction)PyState_get_bag, METH_NOARGS, "" },
    {"copy", (PyCFunction)PyState_copy, METH_NOARGS, "" },
    {"transitions", (PyCFunction)PyState_transitions, METH_VARARGS, "" },
    {"holded", (PyCFunction)PyState_hold, METH_NOARGS, "" },
    {NULL}  /* Sentinel */
};


extern PyTypeObject PyState_Type;


PyDoc_STRVAR(PyState_doc, "State()\
\
Create State");