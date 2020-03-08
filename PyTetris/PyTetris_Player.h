#pragma once

#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

#include "Player.h"
#include "Analyzer.h"

typedef struct PyPlayer {
    PyPlayer() {}
    PyObject_HEAD;
    Player player;
};

void PyPlayer_dealloc(PyPlayer* self);
PyObject* PyPlayer_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int PyPlayer_init(PyPlayer* self, PyObject* args, PyObject* kwds);


PyObject* PyPlayer_has_dest(PyPlayer* self, PyObject* Py_UNUSED(ignored));
PyObject* PyPlayer_set_dest(PyPlayer* self, PyObject* args);
PyObject* PyPlayer_hold(PyPlayer* self, PyObject* Py_UNUSED(ignored));
PyObject* PyPlayer_set_speed(PyPlayer* self, PyObject* args);

static PyMemberDef PyPlayer_members[] = {
    {NULL}  /* Sentinel */
};

PyObject* PyPlayer_tick(PyPlayer* self, PyObject* Py_UNUSED(ignored));

static PyMethodDef PyPlayer_methods[] = {
    {"tick", (PyCFunction)PyPlayer_tick, METH_NOARGS,
     "function that should be called every tick"
    },
    {"has_dest", (PyCFunction)PyPlayer_has_dest, METH_NOARGS,
    "Return if Player has destination spot"
    },
    {"set_dest", (PyCFunction)PyPlayer_set_dest, METH_VARARGS,
    "Set player destination spot"
    },
    {"set_speed", (PyCFunction)PyPlayer_set_speed, METH_VARARGS,
    "Set player falling speed"
    },
    {NULL}  /* Sentinel */
};


static PyTypeObject PyPlayer_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "PyTetris.Player",
    .tp_basicsize = sizeof(PyPlayer),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyPlayer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "AutoPlayer",
    .tp_methods = PyPlayer_methods,
    .tp_members = PyPlayer_members,
    .tp_init = (initproc)PyPlayer_init,
    .tp_new = PyPlayer_new,
};

PyDoc_STRVAR(PyPlayer_doc, "Player()\
\
Create Player");