#pragma once

#include <torch/torch.h>

namespace NeuralNetwork {

class NetworkImpl : public torch::nn::Module
{
public:
  /*
   * Constructor which creates a new neural network instance with the given number of input and output nodes.
   * Also using the given definition of the hidden layers. Each value in the hiddenLayers vector
   * defines a new hidden layer with the corresponding number of nodes.
   */
  NetworkImpl(uint32_t numberOfInputNodes, uint32_t numberOfOutputNode, std::vector<uint32_t> const& hiddenLayers);

public:
  /*
   * Infers the output tensor with the given input tensor.
   */
  [[nodiscard]]
  torch::Tensor forward(torch::Tensor x);

private:
  /*
   * Adds a layer to the neural network with the given parameters.
   */
  void addLayer(size_t layerNumber, uint32_t numberOfInputNodes, uint32_t numberOfOutputNodes);

private:
  std::vector<torch::nn::Sequential> layers{};
};

/*
 * This macro defines the Network class depending on the NetworkImpl class.
 */
TORCH_MODULE(Network);

}
