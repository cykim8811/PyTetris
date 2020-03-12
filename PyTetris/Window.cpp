
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyTetris_Array_API
#include "Window.h"

void Window::draw_block(int x, int y, int type, int size) {

	Block* block = &Tile[type][0];
	int offset_x = x - int(block->size * size / 2),
		offset_y = y - int(block->size * size / 2);
	switch (type)
	{
	case 0:
		offset_y -= int(size / 2);
		break;
	case 1:
	case 2:
	case 4:
	case 5:
	case 6:
		offset_y += int(size / 2);
		break;
	default:
		break;
	}
	for (int ix = 0; ix < block->size; ix++) {
		for (int iy = 0; iy < block->size; iy++) {
			if (!block->at(ix, iy)) continue;
			const int d = type;
			SDL_SetRenderDrawColor(renderer, TileColor[d][0], TileColor[d][1], TileColor[d][2], 255);
			SDL_Rect rect{ offset_x + size * ix, offset_y + size * iy, size, size };
			SDL_RenderFillRect(renderer, &rect);
			SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
			rect = { offset_x + size * ix, offset_y + size * iy, size + 1, size + 1 };
			SDL_RenderDrawRect(renderer, &rect);
		}
	}
}

Window::Window() {
	window = SDL_CreateWindow(
		"PyTetris",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		32 * 22, 32 * 22, 0
	);
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

	prevTime = SDL_GetTicks();
	currTime = 0;

	// TODO: Initialize variables
	key_delay[0] = -1;
	key_delay[1] = -1;
	key_delay[2] = -1;

	falling_pos = { 4, -2, 0 };
	falling_type = pop_next_block();
	for (int i = 0; i < 5; i++) {
		block_next.push_back(pop_next_block());
	}

	block_hold = -1;
	ghost = falling_pos;
	block_update();
	current_gravity_delay = gravity_delay;
}

Window::~Window() {
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
}

bool Window::tick() {
	currTime = SDL_GetTicks();
	int dT = (currTime - prevTime); // ms
	prevTime = currTime;

	for (int key = 0; key < 3; key++) {
		if (key_delay[key] < 0) continue;
		if ((key_delay[key] - dT) < 0) {
			key_delay[key] = ARR;
			key_input(key);
		}
		else {
			key_delay[key] -= dT;
		}
	}

	current_gravity_delay -= dT;
	if (current_gravity_delay < 0) {
		current_gravity_delay = gravity_delay;
		gravity();
	}

	SDL_Event ev;
	while (SDL_PollEvent(&ev)) {
		handle_event(ev);
	}
	if (!running) {
		return false;
		this->~Window();
	};
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
	SDL_RenderClear(renderer);
	draw();
	SDL_RenderPresent(renderer);
	return true;
}

void Window::handle_event(SDL_Event ev) {
	switch (ev.type) {
	case SDL_KEYDOWN:
		if (allow_input) {
			if (ev.key.repeat) return;
			switch (ev.key.keysym.sym) {
			case SDLK_LEFT:
				key_delay[TK_LEFT] = DAS;
				key_input(TK_LEFT);
				break;
			case SDLK_RIGHT:
				key_delay[TK_RIGHT] = DAS;
				key_input(TK_RIGHT);
				break;
			case SDLK_DOWN:
				key_delay[TK_DOWN] = ARR;
				key_input(TK_DOWN);
				break;
			case SDLK_UP:
			case SDLK_x:
				key_input(TK_SPIN);
				break;
			case SDLK_z:
			case SDLK_LCTRL:
				key_input(TK_REVERSED_SPIN);
				break;
			case SDLK_SPACE:
				key_input(TK_DROP);
				break;
			case SDLK_c:
			case SDLK_LSHIFT:
				key_input(TK_HOLD);
				break;
			default:
				break;
			}
		}
		switch (ev.key.keysym.sym) {
		case SDLK_F1:
			break;
		default:
			break;
		}
		break;
	case SDL_KEYUP:
		switch (ev.key.keysym.sym) {
		case SDLK_LEFT:
			key_delay[TK_LEFT] = -1;
			break;
		case SDLK_RIGHT:
			key_delay[TK_RIGHT] = -1;
			break;
		case SDLK_DOWN:
			key_delay[TK_DOWN] = -1;
			break;
		}
		break;
	case SDL_QUIT:
		running = false;
		break;
	default:
		break;
	}
}

void Window::draw() {

	SDL_Rect rect;

	for (int x = 0; x < map.w; x++) {
		for (int y = 0; y < map.h; y++) {
			int d = map.at(x, y);
			if (!d) continue;
			SDL_SetRenderDrawColor(renderer, TileColor[d - 1][0], TileColor[d - 1][1], TileColor[d - 1][2], 255);
			rect = SDL_Rect{ 32 * (6 + x), 32 * (1 + y), 32, 32 };
			SDL_RenderFillRect(renderer, &rect);
		}
	}

	if (show_ghost) {
		Block* ghost_block = &Tile[falling_type][ghost.r];
		for (int x = 0; x < ghost_block->size; x++) {
			for (int y = 0; y < ghost_block->size; y++) {
				if (!ghost_block->at(x, y)) continue;
				if (!map.contains(ghost.x + x, ghost.y + y)) continue;
				const int d = falling_type;
				SDL_SetRenderDrawColor(renderer, TileColor[d][0] * 0.6 + 255 * 0.4, TileColor[d][1] * 0.6 + 255 * 0.4, TileColor[d][2] * 0.6 + 255 * 0.4, 255);
				rect = SDL_Rect{ 32 * (6 + ghost.x + x), 32 * (1 + ghost.y + y), 32, 32 };
				SDL_RenderFillRect(renderer, &rect);
			}
		}
	}

	Block* falling_block = &Tile[falling_type][falling_pos.r];
	for (int x = 0; x < falling_block->size; x++) {
		for (int y = 0; y < falling_block->size; y++) {
			if (!falling_block->at(x, y)) continue;
			if (!map.contains(falling_pos.x + x, falling_pos.y + y)) continue;
			const int d = falling_type;
			SDL_SetRenderDrawColor(renderer, TileColor[d][0], TileColor[d][1], TileColor[d][2], 255);
			rect = SDL_Rect{ 32 * (6 + falling_pos.x + x), 32 * (1 + falling_pos.y + y), 32, 32 };
			SDL_RenderFillRect(renderer, &rect);
		}
	}

	SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
	for (int i = 0; i < 9; i++) {
		SDL_RenderDrawLine(renderer, 32 * (7 + i), 32 * 1, 32 * (7 + i), 32 * 21);
	}
	for (int i = 0; i < 19; i++) {
		SDL_RenderDrawLine(renderer, 32 * 6, 32 * (2 + i), 32 * 16, 32 * (2 + i));
	}


	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	rect = { 32 * 6, 32 * 1, 32 * 10, 32 * 20 };
	SDL_RenderDrawRect(renderer, &rect);

	if (block_hold != -1) {
		draw_block(32 * 3, 32 * 3, block_hold, 28);
	}
	rect = { 32 * 1, 32 * 1, 32 * 4, 32 * 4 };
	SDL_RenderDrawRect(renderer, &rect);

	for (int i = 0; i < 5; i++) {
		draw_block(32 * 19, 32 * 1 + 24 * (2 + i * 5), block_next[i], 18);
		rect = { 32 * 19 - 24 * 2, 32 * 1 + 24 * (i * 5), 24 * 4, 24 * 4 };
		SDL_RenderDrawRect(renderer, &rect);
	}
}

void Window::key_input(int key) {
	Pos newpos;
	switch (key)
	{
	case TK_LEFT:
		newpos = { falling_pos.x - 1, falling_pos.y, falling_pos.r };
		if (map.fit(falling_type, newpos)) {
			falling_pos = newpos;
			if (!map.fit(falling_type, Pos{ newpos.x, newpos.y + 1, newpos.r })) {
				current_gravity_delay = gravity_delay;
			}
		}
		block_update();
		break;
	case TK_RIGHT:
		newpos = { falling_pos.x + 1, falling_pos.y, falling_pos.r };
		if (map.fit(falling_type, newpos)) {
			falling_pos = newpos;
			if (!map.fit(falling_type, Pos{ newpos.x, newpos.y + 1, newpos.r })) {
				current_gravity_delay = gravity_delay;
			}
		}
		block_update();
		break;
	case TK_DOWN:
		newpos = { falling_pos.x, falling_pos.y + 1, falling_pos.r };
		if (map.fit(falling_type, newpos)) {
			falling_pos = newpos;
			if (!map.fit(falling_type, Pos{ newpos.x, newpos.y + 1, newpos.r })) {
				current_gravity_delay = gravity_delay;
			}
		}
		current_gravity_delay = gravity_delay;
		block_update();
		break;
	case TK_SPIN:
		newpos = map.rotate(falling_type, falling_pos, 1);
		if (newpos.r == -1) break;
		if (map.fit(falling_type, newpos)) {
			falling_pos = newpos;
			if (!map.fit(falling_type, Pos{ newpos.x, newpos.y + 1, newpos.r })) {
				current_gravity_delay = gravity_delay;
			}
		}
		block_update();
		break;
	case TK_REVERSED_SPIN:
		newpos = map.rotate(falling_type, falling_pos, -1);
		if (newpos.r == -1) break;
		if (map.fit(falling_type, newpos)) {
			falling_pos = newpos;
			if (!map.fit(falling_type, Pos{ newpos.x, newpos.y + 1, newpos.r })) {
				current_gravity_delay = gravity_delay;
			}
		}
		block_update();
		break;
	case TK_DROP:
		newpos = { falling_pos.x, falling_pos.y + 1, falling_pos.r };
		while (map.fit(falling_type, newpos)) {
			falling_pos = newpos;
			newpos = { falling_pos.x, falling_pos.y + 1, falling_pos.r };
		}
		check();
		break;
	case TK_HOLD:
		hold();
		break;
	default:
		break;
	}
}

int Window::pop_next_block() {
	if (bag_of_blocks.size() == 0) {
		for (int i = 0; i < 7; i++) { bag_of_blocks.push_back(i); }
		shuffle(bag_of_blocks.begin(), bag_of_blocks.end(), default_random_engine((unsigned)time(0)));
	}
	int ret = bag_of_blocks.back();
	bag_of_blocks.pop_back();
	return ret;
}

void Window::check() {
	if (!map.put(falling_type, falling_pos)) {
		game_end();
		return;
	}
	falling_type = block_next[0];
	block_next.erase(block_next.begin());
	block_next.push_back(pop_next_block());
	if (falling_type == 3)
		falling_pos = { 4, -2, 0 };
	else
		falling_pos = { 3, -2, 0 };
	hold_used = false;

	for (int y = map.h - 1; y >= 0; y--) {
		bool isall = true;
		for (int x = 0; x < map.w; x++) {
			if (map.at(x, y) == 0) {
				isall = false;
				break;
			}
		}
		if (isall) {
			
			for (int rx = 0; rx < map.w; rx++) {
				for (int ry = y; ry > 0; ry--) {
					map.set(rx, ry, map.at(rx, ry - 1));
				}
			}
			for (int rx = 0; rx < map.w; rx++) {
				map.set(rx, 0, 0);
			}

			y++;
		}
	}
	current_gravity_delay = gravity_delay;
	block_update();
}

void Window::hold() {
	if (hold_used) return;
	if (block_hold == -1) {
		block_hold = falling_type;
		falling_type = block_next[0];
		block_next.erase(block_next.begin());
		block_next.push_back(pop_next_block());
	}
	else {
		int temp = block_hold;
		block_hold = falling_type;
		falling_type = temp;
	}
	if (falling_type == 3)
		falling_pos = { 4, -2, 0 };
	else
		falling_pos = { 3, -2, 0 };
	hold_used = true;
	current_gravity_delay = gravity_delay;
	block_update();
}

void Window::block_update() {

	Pos newpos = falling_pos;
	ghost = falling_pos;
	while (map.fit(falling_type, newpos)) {
		ghost = newpos;
		newpos = { ghost.x, ghost.y + 1, ghost.r };
	}

}

void Window::game_end() {
	printf("Game End\n");
}

void Window::gravity() {
	if (!has_gravity) { return; }
	Pos newpos = { falling_pos.x, falling_pos.y + 1, falling_pos.r };
	if (map.fit(falling_type, newpos)) {
		falling_pos = newpos;
	}
	else {
		check();
	}
}

