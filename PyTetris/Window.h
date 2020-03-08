#pragma once
#include "SDL.h"
#include "Map.h"

#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>

#include "const_variables.h"

using namespace std;

class Window
{
private:
	Uint32 prevTime;
	Uint32 currTime;

	// L, R, D
	int key_delay[3];
	bool running = true;
	int current_gravity_delay;

	Pos ghost;

	void draw_block(int x, int y, int type, int size = 24);

public:
	Window();
	~Window();
	
	vector<int> bag_of_blocks;

	bool allow_input = true;

	// Settings
	int DAS = 133; // ms
	int ARR = 10;  // ms
	int next_n = 5;

	int gravity_delay = 800;

	// Game data
	Map map = Map(10, 20);
	
	Pos falling_pos;
	int falling_type;

	vector<int> block_next;
	int block_hold;

	bool hold_used = false;

	// System
	SDL_Window* window;
	SDL_Renderer* renderer;

	bool tick(); // Should be called each frame
	
	void handle_event(SDL_Event ev);
	void draw();

	void key_input(int key);

	int pop_next_block();

	void check();
	void hold();

	void block_update(); // on block position or rotation update

	void game_end();
	void gravity();
};

