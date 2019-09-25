#pragma once

#include <torch/torch.h>

namespace NeuralNetwork {

class NetworkImpl : public torch::nn::Module
{
public:
  NetworkImpl(uint32_t numberOfInputNodes, uint32_t numberOfOutputNode, const std::vector<uint32_t>& hiddenLayers);

public:
  torch::Tensor forward(torch::Tensor x);

private:
  void addLayer(size_t layerNumber, uint32_t numberOfInputNodes, uint32_t numberOfOutputNodes);

private:
  std::vector<torch::nn::Linear> layers{};
//  torch::nn::Linear fc1{nullptr};
//  torch::nn::Linear fc2{nullptr};
//  torch::nn::Linear fc3{nullptr};
};

TORCH_MODULE(Network);

}
