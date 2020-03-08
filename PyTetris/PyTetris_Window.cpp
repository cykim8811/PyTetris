#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include "PyTetris_Window.h"

void PyWindow_dealloc(PyWindow* self) {
	self->window.~Window();
}

PyObject* PyWindow_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyWindow* self;
	self = (PyWindow*)type->tp_alloc(type, 0);
	new(self) PyWindow();

	return (PyObject*)self;
}

int PyWindow_init(PyWindow* self, PyObject* args, PyObject* kwds) {
	return 0;
}

PyObject* PyWindow_tick(PyWindow* self, PyObject* Py_UNUSED(ignored)) {
	return Py_BuildValue("i", self->window.tick());
}


PyObject* PyWindow_get_screen(PyWindow* self, PyObject* Py_UNUSED(ignored)) {
	npy_intp dims[2] = { self->window.map.w, self->window.map.h };
	return (PyObject*)PyArray_SimpleNewFromData(2, &dims[0], NPY_INT, (void *)self->window.map.data);
}

PyObject* PyWindow_get_falling_type(PyWindow* self, PyObject* Py_UNUSED(ignored)) {
	return Py_BuildValue("i", self->window.falling_type);
}

PyObject* PyWindow_get_falling_pos(PyWindow* self, PyObject* Py_UNUSED(ignored)) {
	Pos p = self->window.falling_pos;
	return Py_BuildValue("iii", p.x, p.y, p.r);
}
PyObject* PyWindow_get_hold_used(PyWindow* self, PyObject* Py_UNUSED(ignored)) {
	return Py_BuildValue("i", self->window.hold_used);
}
PyObject* PyWindow_get_state(PyWindow* self, PyObject* Py_UNUSED(ignored)) {
	PyObject* arg = Py_BuildValue("ii", 10, 20);
	PyState* ret = (PyState*)PyObject_CallObject((PyObject*)&PyState_Type, arg);
	Py_DECREF(arg);

	ret->bag = self->window.bag_of_blocks;
	ret->block_hold = self->window.block_hold;
	ret->block_next.clear();
	ret->block_next = self->window.block_next;
	// ret->btb = self->window.btb;
	// ret->combo = self->window.combo;
	ret->hold_used = self->window.hold_used;
	ret->screen = Map(self->window.map);

	return (PyObject*)ret;
}