#pragma once
#include "Window.h"
#include "Analyzer.h"

class Player
{
private:
	Uint32 currTime;
	Uint32 prevTime;

	vector<Pos> path;
	vector<int> op;
	int current_delay;
public:
	Player();
	Player(Window* _target);
	Window* window;

	int delay;
	Pos target;
	bool has_target;

	int tick();

	bool has_dest();
	bool set_dest(Pos dest);

	bool hold();
};

