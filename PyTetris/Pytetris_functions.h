#pragma once
#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

#include "Analyzer.h"

PyObject* PyTetris_available_spots(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* PyTetris_available_spots_strict(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* get_block(PyObject* self, PyObject* args, PyObject* kwargs);
//PyObject* PyTetris_transitions(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* PyTetris_search_path_op(PyObject* self, PyObject* args, PyObject* kwargs);

static PyMethodDef PyTetris_functions[] = {
    {"available_spots", (PyCFunction)PyTetris_available_spots, METH_VARARGS | METH_KEYWORDS,
     "Get available spots, including ones that are unreachable"
    },
    {"available_spots_strict", (PyCFunction)PyTetris_available_spots_strict, METH_VARARGS | METH_KEYWORDS,
     "Get available spots"
    },
    {"search_path_op", (PyCFunction)PyTetris_search_path_op, METH_VARARGS | METH_KEYWORDS,
     "Get available spots"
    },
    { "get_block", (PyCFunction)get_block, METH_VARARGS | METH_KEYWORDS,
"get_block(type, angle)\
\
Return block with type and angle"},
    { NULL, NULL, 0, NULL } /* marks end of array */
};
