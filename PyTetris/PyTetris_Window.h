#pragma once

#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

#include "Window.h"
#include "Analyzer.h"
#include "PyTetris_State.h"

typedef struct PyWindow {
    PyWindow() {}
    PyObject_HEAD;
    Window window;
};

void PyWindow_dealloc(PyWindow* self);
PyObject* PyWindow_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int PyWindow_init(PyWindow* self, PyObject* args, PyObject* kwds);

static PyMemberDef PyWindow_members[] = {
    {NULL}  /* Sentinel */
};

PyObject* PyWindow_tick(PyWindow* self, PyObject* Py_UNUSED(ignored));
PyObject* PyWindow_get_screen(PyWindow* self, PyObject* Py_UNUSED(ignored));
PyObject* PyWindow_get_falling_type(PyWindow* self, PyObject* Py_UNUSED(ignored));
PyObject* PyWindow_get_falling_pos(PyWindow* self, PyObject* Py_UNUSED(ignored));
PyObject* PyWindow_get_state(PyWindow* self, PyObject* Py_UNUSED(ignored));
PyObject* PyWindow_set_state(PyWindow* self, PyObject* args);
PyObject* PyWindow_set_gravity(PyWindow* self, PyObject* args);
PyObject* PyWindow_set_ghost(PyWindow* self, PyObject* args);
PyObject* PyWindow_key_input(PyWindow* self, PyObject* args);

static PyMethodDef PyWindow_methods[] = {
    {"tick", (PyCFunction)PyWindow_tick, METH_NOARGS,
     "function that should be called every tick"
    },
    {"get_screen", (PyCFunction)PyWindow_get_screen, METH_NOARGS,
    "Return Map as numpy array"
    },
    {"get_falling_type", (PyCFunction)PyWindow_get_falling_type, METH_NOARGS,
    "Return type of falling block"
    },
    {"get_falling_pos", (PyCFunction)PyWindow_get_falling_pos, METH_NOARGS,
    "Return position of falling block"
    },
    {"get_state", (PyCFunction)PyWindow_get_state, METH_NOARGS,
    "Return state"
    },
    {"set_state", (PyCFunction)PyWindow_set_state, METH_VARARGS,
    "set to state"
    },
    {"set_gravity", (PyCFunction)PyWindow_set_gravity, METH_VARARGS,
    "set gravity"
    },
    {"set_ghost", (PyCFunction)PyWindow_set_ghost, METH_VARARGS,
    "set ghost visibility"
    },
    {"key_input", (PyCFunction)PyWindow_key_input, METH_VARARGS,
    "input key"
    },
    {NULL}  /* Sentinel */
};


static PyTypeObject PyWindow_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "PyTetris.Window",
    .tp_basicsize = sizeof(PyWindow),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyWindow_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Window displaying tetris screen",
    .tp_methods = PyWindow_methods,
    .tp_members = PyWindow_members,
    .tp_init = (initproc)PyWindow_init,
    .tp_new = PyWindow_new,
};

PyDoc_STRVAR(PyWindow_doc, "Window()\
\
Create Window");