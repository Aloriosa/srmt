#pragma once
#include <list>
#include <vector>

class Agent
{
public:
    std::pair<int, int> offset;
    std::pair<int, int> cur_position;
    std::pair<int, int> goal;

    explicit Agent():offset({0,0}), cur_position({0,0}), goal({0,0}){}
    explicit Agent(std::pair<int, int> start_, std::pair<int, int> goal_)
    {
        offset = start_;
        cur_position = start_;
        goal = {goal_.first + offset.first, goal_.second + offset.second};
    }
};