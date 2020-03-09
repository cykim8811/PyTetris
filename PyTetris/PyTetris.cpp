#include <vector>
#include <Python.h>
#include <SDL.h>

#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include <structmember.h>
#include "numpy/arrayobject.h"

#include "PyTetris_functions.h"
#include "PyTetris_Window.h"
#include "PyTetris_Player.h"
#include "PyTetris_State.h"
#include "Window.h"
#include "Analyzer.h"
#include "Map.h"


PyDoc_STRVAR(PyTetris_doc, "Tetris Engine for python");


static PyModuleDef PyTetris_def = {
    PyModuleDef_HEAD_INIT,
    "TetrisEngine",
    PyTetris_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_PyTetris() {
    PyObject* module;

    module = PyModule_Create(&PyTetris_def);
    if (module == NULL) return NULL;


    if (PyType_Ready(&PyWindow_Type) < 0) return NULL;
    Py_INCREF(&PyWindow_Type);
    if (PyModule_AddObject(module, "Window", (PyObject*)&PyWindow_Type) < 0) {
        Py_DECREF(&PyWindow_Type);
        Py_DECREF(module);
        return NULL;
    }

    if (PyType_Ready(&PyPlayer_Type) < 0) return NULL;
    Py_INCREF(&PyPlayer_Type);
    if (PyModule_AddObject(module, "Player", (PyObject*)&PyPlayer_Type) < 0) {
        Py_DECREF(&PyPlayer_Type);
        Py_DECREF(module);
        return NULL;
    }

    if (PyType_Ready(&PyState_Type) < 0) return NULL;
    Py_INCREF(&PyState_Type);
    if (PyModule_AddObject(module, "State", (PyObject*)&PyState_Type) < 0) {
        Py_DECREF(&PyState_Type);
        Py_DECREF(module);
        return NULL;
    }

    PyModule_AddFunctions(module, PyTetris_functions);

    PyModule_AddStringConstant(module, "__author__", "cykim");
    PyModule_AddStringConstant(module, "__version__", "0.1.0");
    PyModule_AddIntConstant(module, "year", 2020);

    import_array();

    SDL_Init(SDL_INIT_EVERYTHING);
    return module;
}
