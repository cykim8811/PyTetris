#pragma once

#include <vector>
#include <algorithm>
#include "Map.h"
#include "Space.h"
#include "const_variables.h"

typedef struct Node {
	Pos pos;
	int G, F;
	int parent;
	int operation;
};

typedef struct path_op {
	vector<Pos> path;
	vector<int> operations;
};

bool operator==(const Pos& lhs, const Pos& rhs);
int index(vector<Node>::iterator _begin, vector<Node>::iterator _end, Pos target);
int index(vector<Pos>::iterator _begin, vector<Pos>::iterator _end, Pos target);


path_op search_path_and_op(Map &screen, int type, Pos from, Pos to);

vector<Pos> available_spots(Map &screen, int type);
vector<Pos> available_spots_strict(Map &screen, int type, Pos start);