#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
namespace py = pybind11;

class Config
{
public:
    bool use_static_cost = true;
    bool use_dynamic_cost = true;
    bool reset_dynamic_cost = true;
    int obs_radius = 5;
    int num_threads = 8;
    int seed = 42;
    std::string path_to_weights = "";
};
