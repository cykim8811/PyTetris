#include "Player.h"

Player::Player(Window* _target) {
	window = _target;
	delay = 200; // ms
	target = { 4, 18, 0 };
	has_target = false;
	current_delay = 0;

	prevTime = SDL_GetTicks();
	currTime = 0;
}

Player::Player() {
	window = nullptr;
	delay = 200; // ms
	target = { 4, 18, 0 };
	has_target = false;
	current_delay = 0;

	prevTime = SDL_GetTicks();
	currTime = 0;
}

int Player::tick() {
	currTime = SDL_GetTicks();
	int dT = (currTime - prevTime); // ms
	prevTime = currTime;

	if (current_delay >= 0) {
		current_delay -= dT;
		return 0;
	}
	current_delay = delay;

	if (!has_target) return 0;
	int ind = index(path.begin(), path.end(), window->falling_pos);
	if (ind == -1) {
		printf("out of path\n");
		has_target = false;
		return 0;
	}

	if (ind >= op.size() - 1) {
		has_target = false;
		window->key_input(TK_DROP);
		return 1;
	}
	window->key_input(op[ind + 1]);
	if (op[ind + 1] == TK_DROP) {
		has_target = false;
	}
	return 1;
}

bool Player::set_dest(Pos dest) {

	path_op calc = search_path_and_op(window->map, window->falling_type, window->falling_pos, dest);
	// may crash - cannot discriminate between falling_pos==dest and cannot_reach situation
	if (calc.path.size() == 0) {
		return false;
	}
	path = calc.path;
	op = calc.operations;
	has_target = true;
	return true;
}

bool Player::has_dest() {
	return has_target;
}

bool Player::hold() {
	if (window->hold_used) {
		return false;
	}
	window->key_input(TK_HOLD);
	return true;
}