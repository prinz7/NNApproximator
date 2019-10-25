#include "NeuralNetwork/logic.h"
#include "Utilities/datanormalizator.h"
#include "Utilities/fileparser.h"

#include <chrono>
#include <random>

namespace NeuralNetwork {

bool Logic::performUserRequest(Utilities::ProgramOptions const& user_options)
{
  options = user_options;
  auto dataOpt = Utilities::FileParser::ParseInputFile(options.InputDataFilePath, options.NumberOfInputVariables,
    options.NumberOfOutputVariables, inputFileHeader);
  if (!dataOpt) {
    return false;
  }

  torch::set_num_threads(options.NumberOfThreads);

  if (options.LogScaling) {
    for (auto& [inputTensor, outputTensor] : *dataOpt) {
      (void) inputTensor;
      Utilities::DataNormalizator::ScaleLogarithmic(outputTensor);
    }
  }

  if (options.InputMinMaxFilePath != Utilities::DefaultValues::INPUT_MIN_MAX_FILE_PATH) {
    std::string fileHeader{};
    auto minMaxOpt = Utilities::FileParser::ParseInputFile(options.InputMinMaxFilePath, options.NumberOfInputVariables,
      options.NumberOfOutputVariables, fileHeader);
    if (!minMaxOpt) {
      return false;
    }

    if (minMaxOpt->size() != 2) {
      std::cout << "Error: File with min/max values has the wrong number of data. Expected 2 values for each column, got: " + std::to_string(minMaxOpt->size()) << std::endl;
      return false;
    }

    normalizeWithFileData(*dataOpt, *minMaxOpt);
  } else {
    Utilities::DataNormalizator::Normalize(*dataOpt, minMax, 0.0, 1.0);  // TODO let user control normalization
  }

  if (options.OutputMinMaxFilePath != Utilities::DefaultValues::OUTPUT_MIN_MAX_FILE_PATH) {
    saveMinMaxToFile();
  }

  network = Network{options.NumberOfInputVariables, options.NumberOfOutputVariables, std::vector<uint32_t>{500, 500}}; // TODO fix hardcoded value

  if (options.InputNetworkParameters != Utilities::DefaultValues::INPUT_NETWORK_PARAMETERS) {
    torch::load(network, options.InputNetworkParameters);
  }

  std::pair<DataVector, DataVector> data;

  if (options.ValidateAfterTraining) {
    data = splitData(*dataOpt, 100.0 - options.ValidationPercentage);
  } else {
    data = std::make_pair(*dataOpt, DataVector());
  }

  trainNetwork(data.first);

  network->eval();

  if (options.OutputNetworkParameters != Utilities::DefaultValues::OUTPUT_NETWORK_PARAMETERS) {
    torch::save(network, options.OutputNetworkParameters);
  }

  if (options.OutputValuesFilePath != Utilities::DefaultValues::OUTPUT_VALUE) {
    saveValuesToFile(*dataOpt, options.OutputValuesFilePath);
  }

  if (options.OutputDiffFilePath != Utilities::DefaultValues::OUTPUT_DIFF) {
    saveDiffToFile(*dataOpt, options.OutputDiffFilePath);
  }

  // Output behaviour of network:
  if (options.PrintBehaviour) {
    if (options.ValidateAfterTraining) {
      std::cout << "R2 score (training): " << calculateR2Score(data.first) << std::endl;
      std::cout << "R2 score alternate (training): " << calculateR2ScoreAlternate(data.first) << std::endl;
      std::cout << "R2 score alternate denormalized (training): " << calculateR2ScoreAlternateDenormalized(data.first) << std::endl;
      std::cout << "R2 score (validation): " << calculateR2Score(data.second) << std::endl;
      std::cout << "R2 score alternate (validation): " << calculateR2ScoreAlternate(data.second) << std::endl;
      std::cout << "R2 score alternate denormalized (validation): " << calculateR2ScoreAlternateDenormalized(data.second) << std::endl;
    }
    std::cout << "R2 score (all): " << calculateR2Score(*dataOpt) << std::endl;
    std::cout << "R2 score alternate (all): " << calculateR2ScoreAlternate(*dataOpt) << std::endl;
    std::cout << "R2 score alternate denormalized (all): " << calculateR2ScoreAlternateDenormalized(*dataOpt) << std::endl;

    if (options.ValidateAfterTraining) {
      std::cout << "\nTraining set:" << std::endl;
      outputBehaviour(data.first);
      std::cout << "\nValidation set:" << std::endl;
      outputBehaviour(data.second);
    } else {
      outputBehaviour(*dataOpt);
    }
  }

  if (options.InteractiveMode) {
    performInteractiveMode();
  }

  return true;
}

void Logic::normalizeWithFileData(DataVector& data, DataVector const& fileMinMax)
{
  inputMinMax = MinMaxVector(options.NumberOfInputVariables);
  outputMinMax = MinMaxVector(options.NumberOfOutputVariables);

  for (uint32_t j = 0; j < options.NumberOfInputVariables; ++j) {
    inputMinMax[j].first = fileMinMax[0].first[j].item<TensorDataType>();
    inputMinMax[j].second = fileMinMax[1].first[j].item<TensorDataType>();
  }
  for (uint32_t j = 0; j < options.NumberOfOutputVariables; ++j) {
    outputMinMax[j].first = fileMinMax[0].second[j].item<TensorDataType>();
    outputMinMax[j].second = fileMinMax[1].second[j].item<TensorDataType>();
  }

  for (auto& [inputTensor, outputTensor] : data) {
    Utilities::DataNormalizator::Normalize(inputTensor, inputMinMax, 0.0, 1.0);
    Utilities::DataNormalizator::Normalize(outputTensor, outputMinMax, 0.0, 1.0);
  }
}

void Logic::trainNetwork(DataVector const& data)
{
  if (data.empty()) {
    return;
  }
  DataVector randomlyShuffledData(data);
  auto const& numberOfEpochs = options.NumberOfEpochs;
  torch::optim::SGD optimizer(network->parameters(), options.LearnRate);

//  std::random_device rd;
//  std::mt19937 g(rd());

  auto lastMeanError = calculateMeanError(data);
  auto currentMeanError = lastMeanError;
  auto start = std::chrono::steady_clock::now();

  bool continueTraining = true;
  uint32_t numberOfDeteriorationsInRow = 0;

  for (uint32_t epoch = 1; epoch <= numberOfEpochs || continueTraining; ++epoch) {
//    std::shuffle(randomlyShuffledData.begin(), randomlyShuffledData.end(), g);
    auto elapsed = std::chrono::duration_cast<Utilities::TimeoutDuration>(std::chrono::steady_clock::now() - start);
    auto remaining = ((elapsed / std::max(epoch - 1, 1u)) * (numberOfEpochs - epoch + 1));
    lastMeanError = currentMeanError;
    currentMeanError = calculateMeanError(data);

    if (lastMeanError - currentMeanError < options.Epsilon) {
      ++numberOfDeteriorationsInRow;
      if (numberOfDeteriorationsInRow > options.NumberOfDeteriorations) {
        continueTraining = false;
      }
    } else {
      numberOfDeteriorationsInRow = 0;
    }

    if (options.ShowProgressDuringTraining) {
      if (epoch > numberOfEpochs) {
        std::cout << "\rContinue training. Mean squared error changed from " << lastMeanError << " to " << currentMeanError << " -- epoch: " << epoch;
        std::flush(std::cout);
      } else {
        std::cout << "\rEpoch " << epoch << " of " << numberOfEpochs << ". Current mean squared error: " << currentMeanError << " previous: "
                  << lastMeanError <<
                  " -- Remaining time: " << formatDuration<std::chrono::milliseconds, std::chrono::hours, std::chrono::minutes, std::chrono::seconds>(
          remaining); // TODO better output
        std::flush(std::cout);
      }
    }

    if (elapsed > options.MaxExecutionTime) {
      std::cout << "\nStop execution (timeout)." << std::endl;
      break;
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

  if (options.PrintBehaviour) {
    std::cout << "\nTraining duration: " << formatDuration<std::chrono::milliseconds, std::chrono::hours, std::chrono::minutes, std::chrono::seconds>
      (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)) << std::endl;
  }
}

void Logic::performInteractiveMode()
{
  std::cout << "Interactive mode activated. Quit with 'q'" << std::endl;
  std::string input{};
  auto inTensor = torch::zeros(options.NumberOfInputVariables, TORCH_DATA_TYPE);
  uint32_t currentVariable = 0;

  while (input != "q") {
    std::cout << "Input variable " << currentVariable << ": ";
    std::flush(std::cout);
    std::cin >> input;

    try {
      TensorDataType value = std::stod(input);
      inTensor[currentVariable++] = value;
    } catch (std::exception const&) {
      if (input != "q") {
        std::cout << "'" << input << "' cannot be cast to double. Quit interactive mode with 'q'." << std::endl;
      }
    }

    if (currentVariable >= options.NumberOfInputVariables) {
      Utilities::DataNormalizator::Normalize(inTensor, inputMinMax, 0.0, 1.0);
      auto output = network->forward(inTensor);
      auto dOutputTensor = output.clone();
      Utilities::DataNormalizator::Denormalize(dOutputTensor, outputMinMax, 0.0, 1.0, true);
      if (options.LogScaling) {
        Utilities::DataNormalizator::UnscaleLogarithmic(dOutputTensor);
      }

      std::cout << "Normalized input: ";
      for (uint32_t i = 0; i < options.NumberOfInputVariables; ++i) {
        std::cout << inTensor[i].item<TensorDataType>() << "  ";
      }

      std::cout << "\nNeural network output: ";
      for (uint32_t i = 0; i < options.NumberOfOutputVariables; ++i) {
        std::cout << dOutputTensor[i].item<TensorDataType>() << " (" << output[i].item<TensorDataType>() << ")  ";
      }
      std::cout << std::endl;
      currentVariable = 0;
    }
  }
}

double Logic::calculateMeanError(DataVector const& testData)
{
  double error = 0;
  for (auto const& [x, y] : testData) {
    auto prediction = network->forward(x);
    auto loss = torch::mse_loss(prediction, y);
    error += loss.item<double>();
  }
  return error / testData.size();
}

double Logic::calculateR2Score(DataVector const& testData)
{
  if (testData.empty()) {
    return 1.0;
  }

  if (testData[0].second.size(0) > 1) {
    std::cout << "R2 score not implemented for multidimensional output." << std::endl;
    return 0.0;
  }
  double SQE = 0.0;
  double SQT = 0.0;

  TensorDataType y_cross = 0.0;
  for (auto const& [x, y] : testData) {
    (void) x;
    y_cross += y[0].item<TensorDataType>();
  }
  y_cross /= testData.size();

  for (auto const& [x, y] : testData) {
    auto prediction = network->forward(x);

    SQE += std::pow(prediction[0].item<TensorDataType>() - y_cross, 2.0);
    SQT += std::pow(y[0].item<TensorDataType>() - y_cross, 2.0);
  }

  return SQE / SQT;
}

double Logic::calculateR2ScoreAlternate(DataVector const& testData)
{
  if (testData.empty()) {
    return 1.0;
  }

  if (testData[0].second.size(0) > 1) {
    std::cout << "R2 score not implemented for multidimensional output." << std::endl;
    return 0.0;
  }

  double SQR = 0.0;
  double SQT = 0.0;

  TensorDataType y_cross = 0.0;
  for (auto const& [x, y] : testData) {
    (void) x;
    y_cross += y[0].item<TensorDataType>();
  }
  y_cross /= testData.size();

  for (auto const& [x, y] : testData) {
    auto prediction = network->forward(x);
    TensorDataType yi = y[0].item<TensorDataType>();

    SQR += std::pow(yi - prediction[0].item<TensorDataType>(), 2.0);
    SQT += std::pow(yi - y_cross, 2.0);
  }

  return 1.0 - (SQR / SQT);
}

double Logic::calculateR2ScoreAlternateDenormalized(DataVector const& testData)
{
  if (testData.empty()) {
    return 1.0;
  }

  if (testData[0].second.size(0) > 1) {
    std::cout << "R2 score not implemented for multidimensional output." << std::endl;
    return 0.0;
  }

  double SQR = 0.0;
  double SQT = 0.0;

  TensorDataType y_cross = 0.0;
  for (auto const& [x, y] : testData) {
    (void) x;
    auto yD = y.clone();
    Utilities::DataNormalizator::Denormalize(yD, outputMinMax, 0.0 , 1.0, false);

    if (options.LogScaling) {
      Utilities::DataNormalizator::UnscaleLogarithmic(yD);
    }

    y_cross += yD[0].item<TensorDataType>();
  }
  y_cross /= testData.size();

  for (auto const& [x, y] : testData) {
    auto prediction = network->forward(x);
    auto yD = y.clone();
    Utilities::DataNormalizator::Denormalize(yD, outputMinMax, 0.0 , 1.0, false);
    Utilities::DataNormalizator::Denormalize(prediction, outputMinMax, 0.0 , 1.0, false);

    if (options.LogScaling) {
      Utilities::DataNormalizator::UnscaleLogarithmic(yD);
      Utilities::DataNormalizator::UnscaleLogarithmic(prediction);
    }

    TensorDataType yi = yD[0].item<TensorDataType>();

    SQR += std::pow(yi - prediction[0].item<TensorDataType>(), 2.0);
    SQT += std::pow(yi - y_cross, 2.0);
  }

  return 1.0 - (SQR / SQT);
}

std::pair<DataVector, DataVector> Logic::splitData(DataVector const& inputData, double const trainingPercentage) const
{
  if (trainingPercentage == 0) {
    return std::make_pair(DataVector(), inputData);
  }
  DataVector trainingData{};
  DataVector validationData{};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 100.0);

  for (auto const& entry : inputData) {
    if (dis(gen) <= trainingPercentage) {
      trainingData.push_back(entry);
    } else {
      validationData.push_back(entry);
    }
  }

  return std::make_pair(trainingData, validationData);
}

void Logic::outputBehaviour(DataVector const& data)
{
  for (auto const& [inputTensor, outputTensor] : data) {
    auto prediction = network->forward(inputTensor);
    auto loss = torch::mse_loss(prediction, outputTensor);

    torch::Tensor dInputTensor = inputTensor.clone();
    torch::Tensor dOutputTensor = outputTensor.clone();
    torch::Tensor dPrediction = prediction.clone();

    Utilities::DataNormalizator::Denormalize(dInputTensor, inputMinMax,0.0, 1.0, true);
    Utilities::DataNormalizator::Denormalize(dOutputTensor, outputMinMax, 0.0, 1.0, true);
    Utilities::DataNormalizator::Denormalize(dPrediction, outputMinMax, 0.0 , 1.0, true);

    if (options.LogScaling) {
      Utilities::DataNormalizator::UnscaleLogarithmic(dOutputTensor);
      Utilities::DataNormalizator::UnscaleLogarithmic(dPrediction);
    }

    std::cout << "\nx: ";
    for (uint32_t i = 0; i < options.NumberOfInputVariables; ++i) std::cout << inputTensor[i].item<TensorDataType>() << " (" << dInputTensor[i].item<TensorDataType>() << ") ";
    std::cout << "\ny: ";
    for (uint32_t i = 0; i < options.NumberOfOutputVariables; ++i) std::cout << outputTensor[i].item<TensorDataType>() << " (" << dOutputTensor[i].item<TensorDataType>() << ") ";
    std::cout << "\nprediction: ";
    for (uint32_t i = 0; i < options.NumberOfOutputVariables; ++i) std::cout << prediction[i].item<TensorDataType>() << " (" << dPrediction[i].item<TensorDataType>() << ") ";
    std::cout << "\nloss: " << loss.item<double>() << std::endl;
  }
}

void Logic::saveValuesToFile(DataVector const& data, std::string const& path)
{
  DataVector values(data.size());

  size_t i = 0;
  for (auto const& [inputTensor, outputTensor] : data) {
    (void) outputTensor;
    auto prediction = network->forward(inputTensor);
    torch::Tensor dInputTensor = inputTensor.clone();

    Utilities::DataNormalizator::Denormalize(dInputTensor, inputMinMax,0.0, 1.0, false);
    Utilities::DataNormalizator::Denormalize(prediction, outputMinMax, 0.0 , 1.0, false);

    if (options.LogScaling) {
      Utilities::DataNormalizator::UnscaleLogarithmic(prediction);
    }

    values[i++] = std::make_pair(dInputTensor, prediction);
  }

  Utilities::FileParser::SaveData(values, path, inputFileHeader);
}

void Logic::saveDiffToFile(DataVector const& data, std::string const& path)
{
  DataVector diff(data.size());

  size_t i = 0;
  for (auto const& [inputTensor, outputTensor] : data) {
    auto prediction = network->forward(inputTensor);
    torch::Tensor dInputTensor = inputTensor.clone();
    torch::Tensor dOutputTensor = outputTensor.clone();

    Utilities::DataNormalizator::Denormalize(dInputTensor, inputMinMax,0.0, 1.0, false);
    Utilities::DataNormalizator::Denormalize(dOutputTensor, outputMinMax, 0.0, 1.0, false);
    Utilities::DataNormalizator::Denormalize(prediction, outputMinMax, 0.0 , 1.0, false);

    if (options.LogScaling) {
      Utilities::DataNormalizator::UnscaleLogarithmic(dOutputTensor);
      Utilities::DataNormalizator::UnscaleLogarithmic(prediction);
    }

    diff[i++] = std::make_pair(dInputTensor, calculateDiff(dOutputTensor, prediction));
  }

  Utilities::FileParser::SaveData(diff, path, inputFileHeader);
}

torch::Tensor Logic::calculateDiff(torch::Tensor const& input1, torch::Tensor const& input2) const
{
  auto output = input1.clone();

  for (int64_t i = 0; i < input1.size(0); ++i) {
    output[i] -= input2[i].item<TensorDataType>();
  }

  return output;
}

void Logic::saveMinMaxToFile() const
{
  auto inTensorDefault = torch::zeros(options.NumberOfInputVariables, TORCH_DATA_TYPE);
  auto outTensorDefault = torch::zeros(options.NumberOfOutputVariables, TORCH_DATA_TYPE);
  DataVector data(2);
  data[0] = std::make_pair(inTensorDefault, outTensorDefault);
  data[1] = std::make_pair(inTensorDefault.clone(), outTensorDefault.clone());

  for (uint32_t i = 0; i < options.NumberOfInputVariables; ++i) {
    data[0].first[i] = inputMinMax[i].first;
    data[1].first[i] = inputMinMax[i].second;
  }

  for (uint32_t i = 0; i < options.NumberOfOutputVariables; ++i) {
    data[0].second[i] = outputMinMax[i].first;
    data[1].second[i] = outputMinMax[i].second;
  }

  Utilities::FileParser::SaveData(data, options.OutputMinMaxFilePath, inputFileHeader);
}

}
