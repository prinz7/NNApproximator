#include "NeuralNetwork/logic.h"

#include "Utilities/fileparser.h"

namespace NeuralNetwork {

bool Logic::performUserRequest(const Utilities::ProgramOptions &options)
{
  auto data = Utilities::FileParser::ParseInputFile(options.InputFilePath, options.NumberOfInputVariables, options.NumberOfOutputVariables);
  if (!data) {
    return false;
  }

  Network network {options.NumberOfInputVariables, options.NumberOfOutputVariables, {30, 30}}; // TODO fix hardcoded value
  trainNetwork(network, options.NumberOfEpochs, *data);

  return true;
}

void Logic::trainNetwork(Network& network, uint32_t numberOfEpochs, const DataVector& data)
{
  torch::optim::SGD optimizer(network.parameters(), 0.0001); // TODO fix hardcoded value

  for (size_t epoch = 1; epoch <= numberOfEpochs; ++epoch) {
    for (auto [x, y] : data) {
      optimizer.zero_grad();

      auto prediction = network.forward(x);

//      prediction = prediction.toType(torch::ScalarType::Long);
//      auto target = y.toType(torch::ScalarType::Long);
      auto loss = torch::mse_loss(prediction, y);
//      auto loss = torch::kl_div(prediction, y);
//      auto loss = torch::nll_loss(prediction, y);
      loss.backward();
      optimizer.step();

      if (epoch == numberOfEpochs) {
        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "y: " << y.item<TensorDataType>() << std::endl;
        std::cout << "prediction: " << prediction.item<TensorDataType>() << std::endl;
        std::cout << "loss: " << loss.item<double>() << std::endl << std::endl;
      }
    }
  }
}

}
