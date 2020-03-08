#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include "PyTetris_State.h"
#include "PyTetris_Window.h"

PyState* duplicate(PyState* original) {
	PyObject* arg = Py_BuildValue("ii", original->screen.w, original->screen.h);
	PyState* dup = (PyState*)PyObject_CallObject((PyObject*)&PyState_Type, arg);
	Py_DECREF(arg);
	dup->bag = original->bag;
	dup->block_hold = original->block_hold;
	dup->block_next = original->block_next;
	dup->btb = original->btb;
	dup->combo = original->combo;
	dup->hold_used = original->hold_used;
	std::copy(&original->screen.data[0], &original->screen.data[original->screen.w * original->screen.h],
		&dup->screen.data[0]);
	return dup;
}

PyTypeObject PyState_Type{
	.ob_base = PyObject_HEAD_INIT(NULL)
	.tp_name = "PyTetris.State",
	.tp_basicsize = sizeof(PyState),
	.tp_itemsize = 0,
	.tp_dealloc = (destructor)PyState_dealloc,
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_doc = "State",
	.tp_methods = PyState_methods,
	.tp_members = PyState_members,
	.tp_init = (initproc)PyState_init,
	.tp_new = PyState_new,
};

void PyState_dealloc(PyState* self) {
}

PyObject* PyState_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyState* self;
	self = (PyState*)type->tp_alloc(type, 0);
	printf("PyState alloc\n");
	new(self) PyState();
	return (PyObject*)self;
}

int PyState_init(PyState* self, PyObject* args, PyObject* kwds) {
	int w = 10, h = 20;
	if (!PyArg_ParseTuple(args, "|ii", &w, &h)) {
		return NULL;
	}
	self->screen.allocate(w, h);
	printf("PyState init\n");
	return 0;
}

PyObject* PyState_get_block_next(PyState* self, PyObject* args) {
	int index;
	if (!PyArg_ParseTuple(args, "i", &index)) {
		return NULL;
	}
	return Py_BuildValue("i", self->block_next[index]);
}

PyObject* PyState_set_block_next(PyState* self, PyObject* args) {
	int index, value;
	if (!PyArg_ParseTuple(args, "ii", &index, &value)) {
		return NULL;
	}
	self->block_next[index] = value;
	Py_RETURN_NONE;
}

PyObject* PyState_get_screen(PyState* self, PyObject* Py_UNUSED(ignore)) {
	PyObject* target;
	npy_intp dims[2] = { self->screen.w, self->screen.h };
	target = PyArray_SimpleNewFromData(2, &dims[0], NPY_INT, self->screen.data);
	return target;
}

PyObject* PyState_set_screen(PyState* self, PyObject* args) {
	PyArrayObject* map;
	if (!PyArg_ParseTuple(args, "O", &map)) {
		return NULL;
	}
	if (map->nd != 2)
		return NULL;
	
	self->screen.~Map();
	self->screen = Map((PyObject*)map);
	Py_RETURN_NONE;
}

PyObject* PyState_get_bag(PyState* self, PyObject* Py_UNUSED(ignore)) {
	PyObject* target;
	npy_intp dims[1] = { self->bag.size() };
	target = PyArray_SimpleNewFromData(1, &dims[0], NPY_INT, &self->bag[0]);
	return target;
}

PyObject* PyState_set_bag(PyState* self, PyObject* args) {
	PyArrayObject* map;
	if (!PyArg_ParseTuple(args, "O", &map)) {
		return NULL;
	}
	if (map->nd != 1)
		return NULL;
	self->bag.clear();
	for (int i = 0; i < map->dimensions[0]; i++) {
		self->bag.push_back(*((int*)map->data + i));
	}

	Py_RETURN_NONE;
}

PyObject* PyState_copy(PyState* self, PyObject* Py_UNUSED(ignore)) {
	return (PyObject*)duplicate(self);
}

PyObject* PyState_transitions(PyState* self, PyObject* Py_UNUSED(ignore)) {
	return (PyObject*) duplicate(self);
}