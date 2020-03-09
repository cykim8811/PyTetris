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
	dup->screen.~Map();
	dup->screen = Map(original->screen.w, original->screen.h);
	dup->screen.data = original->screen.data;
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
	new(self) PyState();
	return (PyObject*)self;
}

int PyState_init(PyState* self, PyObject* args, PyObject* kwds) {
	int w = 10, h = 20;
	if (!PyArg_ParseTuple(args, "|ii", &w, &h)) {
		return NULL;
	}
	self->screen.allocate(w, h);
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
	target = PyArray_SimpleNewFromData(2, &dims[0], NPY_INT, &self->screen.data[0]);
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

bool anyall(int* target) {
	bool ret = false;
	for (int i = 0; i < 200; i++) {
		if (target[i]) {
			ret = true;
			break;
		}
	}
	return ret;
}

PyObject* PyState_transitions(PyState* self, PyObject* args) {
	int bx = 3,
		by = -2,
		br = 0; // TODO: bx = 4 when type == 3
	if (!PyArg_ParseTuple(args, "|iii", &bx, &by, &br)) {
		return NULL;
	}
	PyObject* ret = PyList_New(0);
	vector<Pos> pos_list = available_spots(self->screen, self->block_next[0]);
	for (int i = 0; i < pos_list.size(); i++) {
		Pos c_pos = pos_list[i];
		path_op c_path = search_path_and_op(self->screen, self->block_next[0], Pos{ bx, by, br }, c_pos);
		if (c_path.path.size() == 0) continue;

		// Create new substate
		int score = 0;
		PyState* c_state = duplicate(self);
		c_state->screen.put(self->block_next[0], c_pos);

		int clear_count = 0;

		for (int cy = self->screen.h - 1; cy >= 0; cy--) {
			bool full = true;
			for (int cx = 0; cx < self->screen.w; cx++) {
				if (!c_state->screen.at(cx, cy)) {
					full = false;
					break;
				}
			}
			if (full) {
				for (int cx = 0; cx < self->screen.w; cx++) {
					for (int m = cy; m > 0; m--) {
						c_state->screen.set(cx, m, c_state->screen.at(cx, m - 1));
					}
					c_state->screen.set(cx, 0, 0);
				}
				cy++;
				clear_count++;
			}
		}
		if (clear_count > 0) { // when line cleared

			// Combo Bonus
			if (c_state->combo < 12) { score += combo_score[c_state->combo]; }
			else if (c_state->combo >= 12) { score += 5; }
			c_state->combo += 1;

			// Multiple Line Bonus
			score += clear_score[clear_count];

			// T-spin detection
			if (self->block_next[0] == 5 && (c_path.operations.back() == TK_SPIN || c_path.operations.back() == TK_REVERSED_SPIN)) {
				int block_count = 0;
				if (c_pos.x == -1 || self->screen.at(c_pos.x, c_pos.y)) block_count++;
				if (c_pos.x == -1 || c_pos.y == self->screen.h - 2 || self->screen.at(c_pos.x, c_pos.y + 2)) block_count++;
				if (c_pos.x == self->screen.w - 2 || self->screen.at(c_pos.x + 2, c_pos.y)) block_count++;
				if (c_pos.x == self->screen.w - 2 || c_pos.y == self->screen.h - 2 || self->screen.at(c_pos.x + 2, c_pos.y + 2)) block_count++;

				if (block_count >= 3) { // Detect T-spin
					if (self->btb) {
						score += 1; // Back to back Bonus
					}
					c_state->btb = true;
					if (self->screen.fit(self->block_next[0], Pos{ c_pos.x + 1, c_pos.y, c_pos.r }) ||
						self->screen.fit(self->block_next[0], Pos{ c_pos.x - 1, c_pos.y, c_pos.r }) ||
						self->screen.fit(self->block_next[0], Pos{ c_pos.x, c_pos.y + 1, c_pos.r }) ||
						self->screen.fit(self->block_next[0], Pos{ c_pos.x, c_pos.y - 1, c_pos.r })
						) { // T-spin mini
						score += 0; // tspin mini 
					}
					else {
						score += clear_count * 2;
					}
				}
			}

			// Tetris detection
			if (clear_count == 4) {
				if (self->btb) {
					score += 1; // Back to back Bonus
				}
				c_state->btb = true;
			}

			// Perfect Clear detection
			bool empty = true;
			for (int cy = 0; cy <self->screen.h; cy++) {
				for (int cx = 0; cx < self->screen.w; cx++) {
					if (self->screen.at(cx, cy)) {
						empty = false;
						break;
					}
				}
			}
			if (empty) {
				score += 10;
			}
		}
		else {
			c_state->combo = 0;
		}

		// change state
		c_state->block_next.erase(c_state->block_next.begin());
		if (c_state->bag.size() == 0) {
			for (int i = 0; i < 7; i++) c_state->bag.push_back(i);
			shuffle(c_state->bag.begin(), c_state->bag.end(), default_random_engine((unsigned)time(0)));
		}
		c_state->block_next.push_back(c_state->bag.front());
		c_state->bag.erase(c_state->bag.begin());
		PyObject* op = Py_BuildValue("iii", c_pos.x, c_pos.y, c_pos.r);
		PyObject* tup = Py_BuildValue("OOi", c_state, op, score);
		Py_DECREF(c_state);
		Py_DECREF(op);
		PyList_Append(ret, tup);
		Py_DECREF(tup);
	}
	return ret;
}