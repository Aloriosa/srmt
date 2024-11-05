#pragma once
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <chrono>
#include <iostream>
using AllocatedStringPtr = std::unique_ptr<char, Ort::detail::AllocatedFree>;

class NN_module
{
public:
    std::shared_ptr<Ort::Env> mEnv;
    std::shared_ptr<Ort::Session> mSession;
    Ort::MemoryInfo memoryInfo;

    std::vector<AllocatedStringPtr> input_names_store;
    std::vector<const char*> input_names;
    std::vector<std::vector<int64_t>> input_shapes;

    std::vector<AllocatedStringPtr> output_names_store;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> output_shapes;
    double total_time;
    int obs_radius;

    NN_module(): memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {}
    explicit NN_module(const std::string& path_to_model): memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault))
    {
        std::string instanceName{"Test inference"};
        mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL, instanceName.c_str());
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        //sessionOptions.EnableProfiling("290k_OPT");
        mSession = std::make_shared<Ort::Session>(*mEnv, path_to_model.c_str(), sessionOptions);

        total_time = 0;

        Ort::AllocatorWithDefaultOptions allocator;
        for(size_t n = 0; n < mSession->GetInputCount(); n++)
        {
            auto input_name = mSession->GetInputNameAllocated(n, allocator);
            input_names_store.push_back(std::move(input_name));
            input_names.push_back(input_names_store.back().get());
            Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(n);
            std::vector<int64_t> temp_shape;
            for (auto v: inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape())
                temp_shape.push_back(v);
            input_shapes.push_back(std::move(temp_shape));
        }

        obs_radius = input_shapes[0].back() / 2;

        for(size_t n = 0; n < mSession->GetOutputCount(); n++)
        {
            auto output_name = mSession->GetOutputNameAllocated(n, allocator);
            output_names_store.push_back(std::move(output_name));
            output_names.push_back(output_names_store.back().get());
            Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(n);
            std::vector<int64_t> temp_shape;
            for (auto v: outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape())
                temp_shape.push_back(v);
            output_shapes.push_back(std::move(temp_shape));
        }


    }

    std::pair<std::vector<float>, float> get_output(std::vector<std::vector<float>> input)
    {
        std::vector<Ort::Value> inputTensors;
        for(size_t k = 0; k < input_shapes.size(); k++)
            inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, input[k].data(), input[k].size(), input_shapes[k].data(), input_shapes[k].size()));
        std::vector<std::vector<float>> outputTensorValues;
        for(auto shape:output_shapes)
        {
            int output_size(1);
            for(auto s:shape)
                output_size *= s;
            outputTensorValues.push_back(std::vector<float>(output_size, 0));
        }
        std::vector<Ort::Value> outputTensors;
        for(size_t k = 0; k < output_shapes.size(); k++)
            outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[k].data(), outputTensorValues[k].size(), output_shapes[k].data(), output_shapes[k].size()));
        auto t1 = std::chrono::high_resolution_clock::now();
        mSession->Run(Ort::RunOptions{nullptr}, input_names.data(), inputTensors.data(), inputTensors.size(), output_names.data(),outputTensors.data(), outputTensors.size());
        auto t2 = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        float *floatarr = outputTensors[1].GetTensorMutableData<float>();
        std::pair<std::vector<float>, float> result;
        float total_p(0);
        for(size_t i = 0; i < 5; i++) {
            result.first.push_back(std::exp(floatarr[i]));
            total_p += result.first.back();
        }
        for(size_t i = 0; i < 5; i++)
            result.first[i] /= total_p;
        result.second = outputTensors[3].GetTensorMutableData<float>()[0];
        return result;
    }
};

