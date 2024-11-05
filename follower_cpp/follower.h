#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <random>
#include "NN_module.h"
#include "planner.h"
#include "config.h"
#include "agent.h"
#include "BS_thread_pool.hpp"

namespace py = pybind11;

class Follower {
    std::vector<Agent> agents;
    std::vector<std::vector<int>> agents_pos;
    std::vector<std::vector<int>> grid;
    NN_module actor;
    Config cfg;
    std::vector<std::mt19937> generators;
    std::vector<planner> planners;
    BS::thread_pool pool;

    std::list<std::pair<int, int>> get_occupied_cells(int agent_idx);
    std::vector<float> generate_input(int a_id, int radius, std::list<std::pair<int, int>> path);
    int get_action(size_t agent_idx, std::pair<int, int> cur_position, std::pair<int, int> cur_goal);

public:
    explicit Follower(): pool(BS::thread_pool(cfg.num_threads)){}
    void init(const Config& cfg_, py::array_t<double> array, std::vector<std::pair<int, int>> abs_starts);
    std::vector<int> act(std::vector<std::pair<int, int>> cur_positions, std::vector<std::pair<int, int>> goals);
};

