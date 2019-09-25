#include "NeuralNetwork/logic.h"
#include "Utilities/datanormalizator.h"
#include "Utilities/fileparser.h"

#include <chrono>
#include <random>

namespace NeuralNetwork {

bool Logic::performUserRequest(const Utilities::ProgramOptions& user_options)
{
  options = user_options;
  auto data = Utilities::FileParser::ParseInputFile(options.InputDataFilePath, options.NumberOfInputVariables, options.NumberOfOutputVariables);
  if (!data) {
    return false;
  }

  auto minMax = std::make_pair(MinMaxVector(), MinMaxVector());
  MinMaxVector& inputMinMax = minMax.first;
  MinMaxVector& outputMinMax = minMax.second;

  Utilities::DataNormalizator::Normalize(*data, minMax, 0.0, 1.0);  // TODO let user control normalization

  Network network{options.NumberOfInputVariables, options.NumberOfOutputVariables, std::vector<uint32_t>{4000}}; // TODO fix hardcoded value

  if (options.InputNetworkParameters != Utilities::DefaultValues::INPUT_NETWORK_PARAMETERS) {
    torch::load(network, options.InputNetworkParameters);
  }

  trainNetwork(network, *data);

  network->eval();

  // Output behaviour of network: // TODO user parametrization
  for (auto const& [inputTensor, outputTensor] : *data) {
    auto prediction = network->forward(inputTensor);
    auto loss = torch::mse_loss(prediction, outputTensor);

    torch::Tensor dInputTensor = inputTensor.clone();
    torch::Tensor dOutputTensor = outputTensor.clone();
    torch::Tensor dPrediction = prediction.clone();

    Utilities::DataNormalizator::Denormalize(dInputTensor, inputMinMax,0.0, 1.0, true);
    Utilities::DataNormalizator::Denormalize(dOutputTensor, outputMinMax, 0.0, 1.0, true);
    Utilities::DataNormalizator::Denormalize(dPrediction, outputMinMax, 0.0 , 1.0, true);

    std::cout << "\nx: ";
    for (uint32_t i = 0; i < options.NumberOfInputVariables; ++i) std::cout << inputTensor[i].item<TensorDataType>() << " (" << dInputTensor[i].item<TensorDataType>() << ") ";
    std::cout << "\ny: ";
    for (uint32_t i = 0; i < options.NumberOfOutputVariables; ++i) std::cout << outputTensor[i].item<TensorDataType>() << " (" << dOutputTensor[i].item<TensorDataType>() << ") ";
    std::cout << "\nprediction: ";
    for (uint32_t i = 0; i < options.NumberOfOutputVariables; ++i) std::cout << prediction[i].item<TensorDataType>() << " (" << dPrediction[i].item<TensorDataType>() << ") ";
    std::cout << "\nloss: " << loss.item<double>() << std::endl;
  }

  if (options.OutputNetworkParameters != Utilities::DefaultValues::OUTPUT_NETWORK_PARAMETERS) {
    torch::save(network, options.OutputNetworkParameters);
  }

  return true;
}

void Logic::trainNetwork(Network& network, const DataVector& data)
{
  DataVector randomlyShuffledData(data);
  const auto& numberOfEpochs = options.NumberOfEpochs;
  torch::optim::SGD optimizer(network->parameters(), 0.000001); // TODO fix hardcoded value

  std::random_device rd;
  std::mt19937 g(rd());

  auto lastMeanError = calculateMeanError(network, data);
  auto currentMeanError = lastMeanError;
  auto start = std::chrono::steady_clock::now();

  bool continueTraining = true;
  int32_t numberOfDeteriorationsInRow = 0;

  for (size_t epoch = 1; epoch <= numberOfEpochs || continueTraining; ++epoch) {
    std::shuffle(randomlyShuffledData.begin(), randomlyShuffledData.end(), g);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    auto remaining = ((elapsed / std::max(epoch - 1, 1ul)) * (numberOfEpochs - epoch + 1));
    lastMeanError = currentMeanError;
    currentMeanError = calculateMeanError(network, data);

    if (lastMeanError - currentMeanError < 0.0) {
      ++numberOfDeteriorationsInRow;
      if (numberOfDeteriorationsInRow > 3) {
        continueTraining = false;
      }
    } else {
      numberOfDeteriorationsInRow = 0;
    }

    if (epoch > numberOfEpochs) { // TODO fix hardcoded value
      std::cout << "\rContinue training. Mean squared error changed from " << lastMeanError << " to " << currentMeanError << " -- epoch: " << epoch;
      std::flush(std::cout);
    } else {
      std::cout << "\rEpoch " << epoch << " of " << numberOfEpochs << ". Current mean squared error: " << currentMeanError << " previous: " << lastMeanError <<
                " -- Remaining time: " << formatDuration<std::chrono::milliseconds, std::chrono::hours, std::chrono::minutes, std::chrono::seconds>(remaining); // TODO better output + only if wanted
      std::flush(std::cout);
    }

    for (auto const& [x, y] : randomlyShuffledData) {
      auto prediction = network->forward(x);

//      prediction = prediction.toType(torch::ScalarType::Long);
//      auto target = y.toType(torch::ScalarType::Long);
      auto loss = torch::mse_loss(prediction, y);
//      auto loss = torch::kl_div(prediction, y);
//      auto loss = torch::nll_loss(prediction, y);

      optimizer.zero_grad();

      loss.backward();
      optimizer.step();
    }
  }
  std::cout << "\nTraining duration: " << formatDuration<std::chrono::milliseconds, std::chrono::hours, std::chrono::minutes, std::chrono::seconds>
    (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start));
}

double Logic::calculateMeanError(Network& network, const DataVector &testData)
{
  double error = 0;
  for (auto const& [x, y] : testData) {
    auto prediction = network->forward(x);
    auto loss = torch::mse_loss(prediction, y);
    error += loss.item<double>();
  }
  return error / testData.size();
}

}
