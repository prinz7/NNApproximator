#pragma once

#include <torch/torch.h>

namespace TorchTest {

class Network : torch::nn::Module
{
public:
  Network();

public:
  torch::Tensor forward(torch::Tensor x);

  std::vector<torch::Tensor> parameters();

private:
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear fc3{nullptr};
};

}
