#include "NeuralNetwork/neuralnetwork.h"
#include "Utilities/constants.h"

namespace NeuralNetwork {

NetworkImpl::NetworkImpl(const uint32_t numberOfInputNodes, const uint32_t numberOfOutputNode, const std::vector<uint32_t>& hiddenLayers)
{
  if (hiddenLayers.empty()) {
    addLayer(0, numberOfInputNodes, numberOfOutputNode);
  } else {
    addLayer(0, numberOfInputNodes, hiddenLayers[0]);

    for (size_t i = 1; i < hiddenLayers.size(); ++i) {
      addLayer(i, hiddenLayers[i - 1], hiddenLayers[i]);
    }

    addLayer(hiddenLayers.size(), hiddenLayers[hiddenLayers.size() - 1], numberOfOutputNode);
  }
}

//NetworkImpl::NetworkImpl()
//{
//  fc1 = register_module("fc1", torch::nn::Linear(1, 100));
//  fc2 = register_module("fc2", torch::nn::Linear(100, 100));
//  fc3 = register_module("fc3", torch::nn::Linear(100, 1));
//}

torch::Tensor NetworkImpl::forward(torch::Tensor x)
{
//  x = torch::dropout(torch::sigmoid(layers[0]->forward(x)), 0.2, is_training());
  for (size_t i = 0; i < layers.size() - 1; ++i) {
    x = torch::sigmoid(layers[i]->forward(x));
  }
  x = (layers[layers.size() - 1]->forward(x));

//  x = torch::sigmoid(fc1->forward(x));
//  x = torch::sigmoid(fc2->forward(x));
//  x = (fc3->forward(x));
  return x;
}

void NetworkImpl::addLayer(const size_t layerNumber, const uint32_t numberOfInputNodes, const uint32_t numberOfOutputNodes)
{
  layers.emplace_back(register_module("layer" + std::to_string(layerNumber), torch::nn::Linear(numberOfInputNodes, numberOfOutputNodes)));
  layers[layerNumber]->to(TORCH_DATA_TYPE);
}

}
