#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <cmath>
#include <set>
#include <map>
#include <list>
#include <iostream>
#define INF 1000000000
namespace py = pybind11;

struct Node {
    Node(int _i = INF, int _j = INF, float _g = INF, float _h = 0) : i(_i), j(_j), g(_g), h(_h), f(_g+_h){}
    int i;
    int j;
    float g;
    float h;
    float f;
    std::pair<int, int> parent;
    bool operator<(const Node& other) const
    {
        return this->f < other.f or (std::abs(this->f - other.f) < 1e-5 and this->g < other.g);
    }
    bool operator>(const Node& other) const
    {
        return this->f > other.f or (std::abs(this->f - other.f) < 1e-5 and this->g > other.g);
    }
    bool operator==(const Node& other) const
    {
        return this->i == other.i and this->j == other.j;
    }
    bool operator==(const std::pair<int, int> &other) const
    {
        return this->i == other.first and this->j == other.second;
    }
};

class planner
{
    std::pair<int, int> start;
    std::pair<int, int> goal;
    std::pair<int, int> abs_offset;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> OPEN;
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<float>> num_occupations;
    std::vector<std::vector<float>> penalties;
    std::vector<std::vector<float>> h_values;
    std::vector<std::vector<Node>> nodes;
    bool use_static_cost;
    bool use_dynamic_cost;
    bool reset_dynamic_cost;
    inline float h(std::pair<int, int> n)
    {
        //return abs(n.first - goal.first) + abs(n.second - goal.second);
        return h_values[n.first][n.second];
    }
    std::vector<std::pair<int,int>> get_neighbors(std::pair<int, int> node)
    {
        std::vector<std::pair<int,int>> neighbors;
        std::vector<std::pair<int,int>> deltas = {{0,1},{1,0},{-1,0},{0,-1}};
        for(auto d:deltas)
        {
            std::pair<int,int> n(node.first + d.first, node.second + d.second);
            if(grid[n.first][n.second] == 0)
                neighbors.push_back(n);
        }
        return neighbors;
    }
    void compute_shortest_path()
    {
        Node current;
        while(!OPEN.empty() and !(current == goal))
        {
            current = OPEN.top();
            OPEN.pop();
            if(nodes[current.i][current.j].g < current.g)
                continue;
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost(1);
                if(use_static_cost)
                    cost = penalties[n.first][n.second];
                if(use_dynamic_cost)
                    cost += num_occupations[n.first][n.second];
                if(nodes[n.first][n.second].g > current.g + cost)
                {
                    OPEN.push(Node(n.first, n.second, current.g + cost, h(n)));
                    nodes[n.first][n.second].g = current.g + cost;
                    nodes[n.first][n.second].parent = {current.i, current.j};
                }
            }
        }
    }

    float get_avg_distance(int si, int sj)
    {
        std::queue<std::pair<int, int>> fringe;
        fringe.emplace(si, sj);
        auto result = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid.front().size(), -1));
        result[si][sj] = 0;
        std::vector<std::pair<int, int>> moves = {{0,1},{1,0},{-1,0},{0,-1}};
        while(!fringe.empty())
        {
            auto pos = fringe.front();
            fringe.pop();
            for(const auto& move: moves)
            {
                int new_i(pos.first + move.first), new_j(pos.second + move.second);
                if(grid[new_i][new_j] == 0 && result[new_i][new_j] < 0)
                {
                    result[new_i][new_j] = result[pos.first][pos.second] + 1;
                    fringe.emplace(new_i, new_j);
                }
            }
        }
        float avg_dist(0), total_nodes(0);
        for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid[0].size(); j++)
                if(result[i][j] > 0)
                {
                    avg_dist += result[i][j];
                    total_nodes++;
                }
        return avg_dist/total_nodes;
    }

    void update_h_values(std::pair<int, int> g)
    {
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
        h_values = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), INF));
        h_values[g.first][g.second] = 0;
        open.push(Node(g.first, g.second, 0, 0));
        while(!open.empty())
        {
            Node current = open.top();
            open.pop();
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost(1);
                if(use_static_cost)
                    cost = penalties[n.first][n.second];
                if(h_values[n.first][n.second] > current.g + cost)
                {
                    open.push(Node(n.first, n.second, current.g + cost, 0));
                    h_values[n.first][n.second] = current.g + cost;
                }
            }
        }
    }

    void reset()
    {
        nodes = std::vector<std::vector<Node>>(grid.size(), std::vector<Node>(grid.front().size(), Node()));
        OPEN = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>();
        Node s = Node(start.first, start.second, 0, h(start));
        OPEN.push(s);
    }

public:
    planner(std::vector<std::vector<int>> _grid={}, float _use_static_cost=1.0, float _use_dynamic_cost=1.0, bool _reset_dynamic_cost=true):
    grid(_grid), use_static_cost(_use_static_cost), use_dynamic_cost(_use_dynamic_cost), reset_dynamic_cost(_reset_dynamic_cost)
    {
        abs_offset = {0, 0};
        goal = {0,0};
        start = {0, 0};
        nodes = std::vector<std::vector<Node>>(grid.size(), std::vector<Node>(grid.front().size(), Node()));
        num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        penalties = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 1));
    }

    std::vector<std::vector<float>> get_num_occupied_matrix()
    {
        return num_occupations;
    }

    std::vector<std::vector<float>> precompute_penalty_matrix(int obs_radius)
    {
        penalties = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        float max_avg_dist(0);
        for(size_t i = obs_radius; i < grid.size() - obs_radius; i++)
            for(size_t j = obs_radius; j < grid.front().size() - obs_radius; j++)
                if(grid[i][j] == 0)
                {
                    penalties[i][j] = get_avg_distance(i, j);
                    max_avg_dist = std::fmax(max_avg_dist, penalties[i][j]);
                }
        for(size_t i = obs_radius; i < grid.size() - obs_radius; i++)
            for(size_t j = obs_radius; j < grid.front().size() - obs_radius; j++)
                if(grid[i][j] == 0)
                    penalties[i][j] = max_avg_dist / penalties[i][j];
        return penalties;
    }

    void set_penalties(std::vector<std::vector<float>> _penalties)
    {
        penalties = std::move(_penalties);
    }

    void update_occupied_cells(const std::list<std::pair<int, int>>& _occupied_cells, std::pair<int, int> cur_goal)
    {
        if(reset_dynamic_cost)
            if(goal.first != cur_goal.first || goal.second != cur_goal.second)
                num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        for(auto o:_occupied_cells)
            num_occupations[o.first][o.second] += 1.0;
    }

    void update_occupations(py::array_t<double> array, std::pair<int, int> cur_pos, std::pair<int, int> cur_goal)
    {
        cur_goal = {cur_goal.first + abs_offset.first, cur_goal.second + abs_offset.second};
        if(reset_dynamic_cost)
            if(goal.first != cur_goal.first || goal.second != cur_goal.second)
                num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        py::buffer_info buf = array.request();
        std::list<std::pair<int, int>> occupied_cells;
        double *ptr = (double *) buf.ptr;
        cur_pos = {cur_pos.first + abs_offset.first, cur_pos.second + abs_offset.second};
        for(size_t i = 0; i < static_cast<size_t>(buf.shape[0]); i++)
            for(size_t j = 0; j < static_cast<size_t>(buf.shape[1]); j++)
                if(ptr[i*buf.shape[1] + j] != 0)
                    occupied_cells.push_back({cur_pos.first + i, cur_pos.second + j});
        for(auto o:occupied_cells)
            num_occupations[o.first][o.second]+= 1.0;
    }

    void update_path(std::pair<int, int> s, std::pair<int, int> g)
    {
        s = {s.first + abs_offset.first, s.second + abs_offset.second};
        g = {g.first + abs_offset.first, g.second + abs_offset.second};
        start = s;
        if(goal != g)
            update_h_values(g);
        goal = g;
        reset();
        compute_shortest_path();
    }

    std::list<std::pair<int, int>> get_path()
    {
        std::list<std::pair<int, int>> path;
        std::pair<int, int> next_node(INF,INF);
        if(nodes[goal.first][goal.second].g < INF)
            next_node = goal;
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
        {
            while (nodes[next_node.first][next_node.second].parent != start) {
                path.push_back(next_node);
                next_node = nodes[next_node.first][next_node.second].parent;
            }
            path.push_back(next_node);
            path.push_back(start);
            path.reverse();
        }
        for(auto it = path.begin(); it != path.end(); it++)
        {
            it->first -= abs_offset.first;
            it->second -= abs_offset.second;
        }
        return path;
    }
    std::pair<std::pair<int, int>, std::pair<int, int>> get_next_node()
    {
        std::pair<int, int> next_node(INF, INF);
        if(nodes[goal.first][goal.second].g < INF)
            next_node = goal;
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
            while (nodes[next_node.first][next_node.second].parent != start)
                next_node = nodes[next_node.first][next_node.second].parent;
        if(next_node == start)
            next_node = {INF, INF};
        if(next_node.first < INF)
            return {{start.first - abs_offset.first, start.second - abs_offset.second},
                    {next_node.first - abs_offset.first, next_node.second - abs_offset.second}};
        return {{INF, INF}, {INF, INF}};
    }
    void set_abs_start(std::pair<int, int> offset)
    {
        abs_offset = offset;
    }
};
