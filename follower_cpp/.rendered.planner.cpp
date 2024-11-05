// cppimport
#include "planner.h"

PYBIND11_MODULE(planner, m) {
    py::class_<planner>(m, "planner")
            .def(py::init<std::vector<std::vector<int>>, bool, bool, bool>())
            .def("set_abs_start", &planner::set_abs_start)
            .def("update_path", &planner::update_path)
            .def("get_path", &planner::get_path)
            .def("get_next_node", &planner::get_next_node)
            .def("precompute_penalty_matrix", &planner::precompute_penalty_matrix)
            .def("set_penalties", &planner::set_penalties)
            .def("update_occupations", &planner::update_occupations);
}

