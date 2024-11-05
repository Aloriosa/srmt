// cppimport
#include "follower.h"

std::list<std::pair<int, int>> Follower::get_occupied_cells(int agent_idx)
{
    std::pair<int, int> cur_pos = agents[agent_idx].cur_position;
    std::list<std::pair<int, int>> result;
    for(auto & agent : agents) {
        int di = std::abs(agent.cur_position.first - cur_pos.first);
        int dj = std::abs(agent.cur_position.second - cur_pos.second);
        if(di <= cfg.obs_radius && dj <= cfg.obs_radius) {
            if(di == 0 && dj == 0)
                continue;
            result.push_back(agent.cur_position);
        }
    }
    return result;
}

std::vector<float> Follower::generate_input(int a_id, int radius, std::list<std::pair<int, int>> path)
{
    int obs_size = radius * 2 + 1;
    std::vector<float> input(obs_size*obs_size*2, 0);
    for(int i = -radius; i <= radius; i++)
        for(int j = -radius; j <= radius; j++)
        {
            if(grid[agents[a_id].cur_position.first + i][agents[a_id].cur_position.second + j])
                input[(i + radius) * obs_size + j + radius] = -1;
            if(agents_pos[agents[a_id].cur_position.first + i][agents[a_id].cur_position.second + j])
                input[(i + radius) * obs_size + j + radius + obs_size * obs_size] = 1;
        }
    if(!path.empty())
    {
        auto cur_pos = path.begin();
        while(cur_pos != path.end() && std::abs(cur_pos->first - agents[a_id].cur_position.first) <= radius && std::abs(cur_pos->second - agents[a_id].cur_position.second) <= radius)
        {
            input[obs_size * (radius + cur_pos->first - agents[a_id].cur_position.first) + radius + cur_pos->second - agents[a_id].cur_position.second] = 1;
            cur_pos++;
        }
    }
    return input;
}

int Follower::get_action(size_t agent_idx, std::pair<int, int> cur_position, std::pair<int, int> cur_goal)
{
    std::list<std::pair<int, int>> occupied_cells = get_occupied_cells(agent_idx);
    planners[agent_idx].update_occupied_cells(occupied_cells, agents[agent_idx].goal);
    planners[agent_idx].update_path(agents[agent_idx].cur_position, agents[agent_idx].goal);
    std::list<std::pair<int, int>> path = planners[agent_idx].get_path();
    auto input = generate_input(agent_idx, actor.obs_radius, path);
    auto result = actor.get_output({input, {0}});
    std::vector<int> score;
    score.reserve(result.first.size());
    for(auto v: result.first)
        score.push_back(static_cast<int>(v * 1e6));
    std::discrete_distribution<int> distr(score.begin(), score.end());
    int action = distr(generators[agent_idx]);
    return action;
}

std::vector<int> Follower::act(std::vector<std::pair<int, int>> cur_positions, std::vector<std::pair<int, int>> goals)
{
    std::vector<int> actions(agents.size(), 0);

    for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
    {
        agents_pos[agents[agent_idx].cur_position.first][agents[agent_idx].cur_position.second] = 0;
        agents[agent_idx].cur_position = {cur_positions[agent_idx].first + agents[agent_idx].offset.first,
                                          cur_positions[agent_idx].second + agents[agent_idx].offset.second};
        agents_pos[agents[agent_idx].cur_position.first][agents[agent_idx].cur_position.second] = 1;
        agents[agent_idx].goal = {goals[agent_idx].first + agents[agent_idx].offset.first,
                                  goals[agent_idx].second + agents[agent_idx].offset.second};
    }
    if(cfg.num_threads == 1)
    {
        for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
            actions[agent_idx] = get_action(agent_idx, cur_positions[agent_idx], goals[agent_idx]);
    }
    else
    {
        BS::multi_future<int> future(agents.size());
        for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
            future[agent_idx] = pool.submit(&Follower::get_action, this, agent_idx, cur_positions[agent_idx], goals[agent_idx]);
        actions = future.get();
    }
    return actions;
}

void Follower::init(const Config& cfg_, py::array_t<double> array, std::vector<std::pair<int, int>> abs_starts)
{
    cfg = cfg_;
    pool.reset(cfg.num_threads);
    actor = NN_module(cfg.path_to_weights);

    py::buffer_info buf = array.request();
    double *ptr = (double *) buf.ptr;
    grid = std::vector<std::vector<int>>(buf.shape[0], std::vector<int>(buf.shape[1], 0));
    agents_pos = grid;
    for(size_t i = 0; i < static_cast<size_t>(buf.shape[0]); i++)
        for(size_t j = 0; j < static_cast<size_t>(buf.shape[1]); j++)
            grid[i][j] = ptr[i*buf.shape[1] + j];

    agents = std::vector<Agent>(abs_starts.size());
    generators = std::vector<std::mt19937>(agents.size());
    planners.clear();
    planner temp_planner(grid, cfg.use_static_cost, cfg.use_dynamic_cost, cfg.reset_dynamic_cost);
    auto pen_matrix = temp_planner.precompute_penalty_matrix(cfg.obs_radius);
    for(size_t agent_idx = 0; agent_idx < agents.size(); agent_idx++)
    {
        agents[agent_idx].offset = abs_starts[agent_idx];
        agents_pos[abs_starts[agent_idx].first][abs_starts[agent_idx].second] = 1;
        planners.emplace_back(grid, cfg.use_static_cost, cfg.use_dynamic_cost, cfg.reset_dynamic_cost);
        planners.back().set_penalties(pen_matrix);
        generators[agent_idx].seed(cfg.seed);
    }
}

PYBIND11_MODULE(follower, m) {
py::class_<Follower>(m, "Follower")
.def(py::init<>())
.def("act", &Follower::act)
.def("init", &Follower::init)
;
}

<%
cfg['libraries'] = ['onnxruntime']
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>