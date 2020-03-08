#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include "PyTetris_Player.h"
#include "PyTetris_Window.h"

void PyPlayer_dealloc(PyPlayer* self) {
	self->player.~Player();
}

PyObject* PyPlayer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyPlayer* self;
	self = (PyPlayer*)type->tp_alloc(type, 0);
	new(self) PyPlayer();

	return (PyObject*)self;
}

int PyPlayer_init(PyPlayer* self, PyObject* args, PyObject* kwds) {
	PyObject* target;
	if (!PyArg_ParseTuple(args, "O", &target)) {
		return NULL;
	}
	self->player.window = &((PyWindow*)target)->window;
	return 0;
}

PyObject* PyPlayer_tick(PyPlayer* self, PyObject* Py_UNUSED(ignored)) {
	return Py_BuildValue("i", self->player.tick());
}

PyObject* PyPlayer_has_dest(PyPlayer* self, PyObject* Py_UNUSED(ignored)) {
	return Py_BuildValue("i", self->player.has_dest());
}

PyObject* PyPlayer_set_dest(PyPlayer* self, PyObject* args) {
	PyObject* destination;
	if (!PyArg_ParseTuple(args, "O", &destination)) {
		return NULL;
	}
	int f_x, f_y, f_r;
	if (!PyArg_ParseTuple(destination, "iii", &f_x, &f_y, &f_r)) {
		return NULL;
	}
	return Py_BuildValue("i", self->player.set_dest(Pos{ f_x, f_y, f_r }));
}

PyObject* PyPlayer_hold(PyPlayer* self, PyObject* Py_UNUSED(ignored)) {
	return Py_BuildValue("i", self->player.hold());
}


PyObject* PyPlayer_set_speed(PyPlayer* self, PyObject* args) {
	if (!PyArg_ParseTuple(args, "i", &self->player.delay)) {
		return NULL;
	}
	Py_RETURN_NONE;
}
