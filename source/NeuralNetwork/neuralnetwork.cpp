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

torch::Tensor NetworkImpl::forward(torch::Tensor x)
{
  for (auto& layer : layers) {
    x = layer->forward(x);
  }

  return x;
}

void NetworkImpl::addLayer(const size_t layerNumber, const uint32_t numberOfInputNodes, const uint32_t numberOfOutputNodes)
{
  layers.emplace_back(register_module("layer" + std::to_string(layerNumber),
    torch::nn::Sequential(torch::nn::Linear(numberOfInputNodes, numberOfOutputNodes))));
  layers[layerNumber]->to(TORCH_DATA_TYPE);
}

}
