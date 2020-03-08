#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API

#include "Analyzer.h"


bool operator==(const Pos& lhs, const Pos& rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.r == rhs.r);
}

int index(vector<Node>::iterator _begin, vector<Node>::iterator _end, Pos target) {
    int i = 0;
    for (vector<Node>::iterator iter = _begin; iter != _end; ++iter) {
        if ((*iter).pos == target) {
            return i;
        }
        i++;
    }
    return -1;
}
int index(vector<Pos>::iterator _begin, vector<Pos>::iterator _end, Pos target) {
    int i = 0;
    for (vector<Pos>::iterator iter = _begin; iter != _end; ++iter) {
        if ((*iter) == target) {
            return i;
        }
        i++;
    }
    return -1;
}

int calc_F(int G, Pos p, Pos to) {
    return G + abs(p.x - to.x)*16 + abs(p.y - to.y) + min((p.r - to.r + 4) % 4, (to.r - p.r + 4) % 4)*8;
}

path_op search_path_and_op(Map &screen, int type, Pos from, Pos to) {
    int w = screen.w,
        h = screen.h;

    Space<bool> map(w + 1, h + 3, 4, -2, -4, 0);

    // Create 3D vector
    for (int x = map.begin(0); x < map.end(0); x++) {
        for (int y = map.begin(1); y < map.end(1); y++) {
            for (int r = 0; r < 4; r++) {
                *map.at(x, y, r) = screen.fit(type, Pos{ x, y, r });
            }
        }
    }

    vector<Node> opened_nodes;
    vector<Node> closed_nodes;


    opened_nodes.push_back(Node{ Pos{from.x, from.y, from.r}, 0,
        (abs(from.x - to.x) + abs(from.y - to.y)), -1, -1 });

    bool finished = false;

    while (!finished) {
        // sort by F
        sort(opened_nodes.begin(), opened_nodes.end(),
            [](const Node& ln, const Node& rn) {
                return ln.F > rn.F;
            });

        if (opened_nodes.size() == 0) return path_op{ {}, {} };
        Node selected = opened_nodes.back();
        opened_nodes.pop_back();

        closed_nodes.push_back(selected);

        int selected_index = closed_nodes.size() - 1;

        if (screen.drop(type, selected.pos) == to) {
            opened_nodes.push_back(Node{ to, 0, 0, selected_index, 5 });
            finished = true;
            break;
        }

        vector<Pos> neighbor;
        neighbor.push_back(Pos{ selected.pos.x - 1, selected.pos.y, selected.pos.r });
        neighbor.push_back(Pos{ selected.pos.x + 1, selected.pos.y, selected.pos.r });
        neighbor.push_back(Pos{ selected.pos.x, selected.pos.y + 1, selected.pos.r });
        neighbor.push_back(screen.rotate(type, selected.pos, 1));
        neighbor.push_back(screen.rotate(type, selected.pos, -1));

        for (int i = 0; i < neighbor.size(); i++) { // i : operation, so neighbor vector should not be shuffled
            Pos c_neigbor = neighbor[i];
            if (!map.contains(c_neigbor.x, c_neigbor.y, c_neigbor.r)) continue;
            if (!*map.at(c_neigbor.x, c_neigbor.y, c_neigbor.r)) continue;
            if (index(closed_nodes.begin(), closed_nodes.end(), c_neigbor) != -1) continue;
            if (c_neigbor == to) {
                opened_nodes.push_back(Node{ c_neigbor, 0, 0, selected_index, i });
                finished = true;
                break;
            }
            int open_ind = index(opened_nodes.begin(), opened_nodes.end(), c_neigbor);
            if (open_ind != -1) {
                if (opened_nodes[open_ind].G > selected.G + 1) {
                    opened_nodes[open_ind].parent = selected_index;
                    opened_nodes[open_ind].G = selected.G + 1;
                    opened_nodes[open_ind].operation = i;
                    opened_nodes[open_ind].F = calc_F(opened_nodes[open_ind].G, opened_nodes[open_ind].pos, to);

                }
            }
            else {
                opened_nodes.push_back(Node{ c_neigbor, selected.G + 1, calc_F(selected.G + 1, c_neigbor, to),
                    selected_index, i});
            }
        }
        if (opened_nodes.size() == 0) return path_op{ {}, {} };
        if (finished) break;
    }

    vector<Pos> ret_node;
    vector<int> ret_oper;

    ret_node.push_back(opened_nodes.back().pos);
    ret_oper.push_back(opened_nodes.back().operation);

    int prev = opened_nodes.back().parent;
    while (prev != -1) {
        ret_node.push_back(closed_nodes[prev].pos);
        ret_oper.push_back(closed_nodes[prev].operation);
        prev = closed_nodes[prev].parent;
    }
    reverse(ret_node.begin(), ret_node.end());
    reverse(ret_oper.begin(), ret_oper.end());

    return path_op{ ret_node, ret_oper };

}

vector<Pos> available_spots(Map &screen, int type) {
    vector<Pos> retlist;
    int rm = 4;
    if (type == 3)
        rm = 1;
    for (int r = 0; r < rm; r++) {
        for (int x = -2; x < screen.w - 2 + 1; x++) {
            bool prev = false;
            for (int y = -2; y < screen.h - 2 + 1; y++) {
                if (screen.fit(type, Pos{ x, y, r })) {
                    prev = true;
                    continue;
                }
                else {
                    if (prev) {
                        retlist.push_back(Pos{ x, y - 1, r });
                    }
                    prev = false;
                }
            }
            if (prev) {
                retlist.push_back(Pos{ x, (int)screen.h - 2, r });
            }
        }
    }
    return retlist;
}

vector<Pos> available_spots_strict(Map &screen, int type, Pos start) {
    vector<Pos> available_list = available_spots(screen, type);
    vector<Pos> retlist;
    for (int i = 0; i < available_list.size(); i++) {
        path_op calc = search_path_and_op(screen, type, start, available_list[i]);
        if (calc.path.size() != 0) {
            retlist.push_back(available_list[i]);
        }
    }
    return retlist;
}
