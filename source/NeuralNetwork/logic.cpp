#include "NeuralNetwork/logic.h"
#include "Utilities/fileparser.h"

#include <chrono>

namespace NeuralNetwork {

bool Logic::performUserRequest(const Utilities::ProgramOptions &options)
{
  auto data = Utilities::FileParser::ParseInputFile(options.InputFilePath, options.NumberOfInputVariables, options.NumberOfOutputVariables);
  if (!data) {
    return false;
  }

  int32_t tryCount = 0;

  while (tryCount < 100) { // TODO fix initialization
    Network network{options.NumberOfInputVariables, options.NumberOfOutputVariables, {1000, 1000}}; // TODO fix hardcoded value
    if (network.forward(data->front().first).item<float>() > 0.0f) {
      std::cout << "Needed " << tryCount + 1 << " tries." << std::endl;
      tryCount = 100;
      trainNetwork(network, options.NumberOfEpochs, *data);
    }
    ++tryCount;
  }

  return true;
}

void Logic::trainNetwork(Network& network, uint32_t numberOfEpochs, const DataVector& data)
{
  torch::optim::SGD optimizer(network.parameters(), 0.0000001); // TODO fix hardcoded value

  auto start = std::chrono::steady_clock::now();
  for (size_t epoch = 1; epoch <= numberOfEpochs; ++epoch) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << "\rEpoch " << epoch << " of " << numberOfEpochs << ". Current mean error: " << calculateMeanError(network, data) <<
                 " -- Remaining time: " << ((elapsed / std::max(epoch - 1, 1ul)) * (numberOfEpochs - epoch + 1)) << " seconds."; // TODO better output + only if wanted
    std::flush(std::cout);

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

double Logic::calculateMeanError(Network &network, const DataVector &testData)
{
  double error = 0;
  for (auto [x, y] : testData) {
    auto prediction = network.forward(x);
    auto loss = torch::mse_loss(prediction, y);
    error += prediction.item<double>();
  }
  return error / testData.size();
}

}
