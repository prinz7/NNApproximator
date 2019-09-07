#include "MachineLearningTest/neuralnetwork.h"

namespace TorchTest {

Network::Network()
{
  fc1 = register_module("fc1", torch::nn::Linear(1, 10));
  fc2 = register_module("fc2", torch::nn::Linear(10, 10));
  fc3 = register_module("fc3", torch::nn::Linear(10, 1));
}

torch::Tensor Network::forward(torch::Tensor x)
{
  x = torch::relu(fc1->forward(x));
  x = torch::relu(fc2->forward(x));
  x = torch::relu(fc3->forward(x));
  return x;
}

std::vector<torch::Tensor> Network::parameters()
{
  return torch::nn::Module::parameters();
}

}
