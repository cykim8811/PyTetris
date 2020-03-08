#pragma once

template <typename T>
class Space {
public:
	int dim[3];
	int offset[3];
	Space(int d1, int d2, int d3, int off_x = 0, int off_y = 0, int off_z = 0);
	~Space();
	T* at(int x, int y, int z);
	bool contains(int x, int y, int z);
	T* data;
	int begin(int x) {
		return offset[x];
	}
	int end(int x) {
		return offset[x] + dim[x];
	}
};


template <typename T>
Space<T>::Space(int d1, int d2, int d3, int off_x, int off_y, int off_z) {
	data = new T[d1 * d2 * d3];
	dim[0] = d1;
	dim[1] = d2;
	dim[2] = d3;
	offset[0] = off_x;
	offset[1] = off_y;
	offset[2] = off_z;
}

template <typename T>
Space<T>::~Space() {
	delete[] data;
}

template <typename T>
T* Space<T>::at(int x, int y, int z) {
	return data + ((x - offset[0]) * dim[1] * dim[2]
		+ (y - offset[1]) * dim[2]
		+ (z - offset[2]));
}

template <typename T>
bool Space<T>::contains(int x, int y, int z) {
	return x >= offset[0] && x < offset[0] + dim[0] &&
		y >= offset[1] && y < offset[1] + dim[1] &&
		z >= offset[2] && z < offset[2] + dim[2];
}
