#include "NeuralNetwork/logic.h"
#include "Utilities/dataprocessor.h"
#include "Utilities/datasplitter.h"
#include "Utilities/fileparser.h"

#include <chrono>
#include <random>

namespace NeuralNetwork {

bool Logic::performUserRequest(Utilities::ProgramOptions const& user_options)
{
  options = user_options;

  if (options.DebugOutput) {
    std::cout << "Read input file..." << std::endl;
  }
  auto dataOpt = Utilities::FileParser::ParseInputFile(options.InputDataFilePath, options.NumberOfInputVariables,
    options.NumberOfOutputVariables, inputFileHeader);
  if (!dataOpt) {
    return false;
  }

  useMixedScaling = options.LogLinScaling || options.LogSqrtScaling;
  torch::set_num_threads(options.NumberOfThreads);

  if (options.RNGSeed) {
    torch::manual_seed(*options.RNGSeed);
  }

  if (options.DebugOutput) {
    std::cout << "Scale the output tensors..." << std::endl;
  }

  if (options.LogScaling) {
    for (auto& [inputTensor, outputTensor] : *dataOpt) {
      (void) inputTensor;
      Utilities::DataProcessor::ScaleLogarithmic(outputTensor);
    }
  } else if (options.SqrtScaling) {
    for (auto& [inputTensor, outputTensor] : *dataOpt) {
      (void) inputTensor;
      Utilities::DataProcessor::ScaleSquareRoot(outputTensor);
    }
  } else if (options.LogLinScaling) {
    for (auto& [inputTensor, outputTensor] : *dataOpt) {
      if (inputTensor[options.MixedScalingInputVariable].item<TensorDataType>() <= options.MixedScalingThreshold) {
        Utilities::DataProcessor::ScaleLogarithmic(outputTensor);
      }
    }
  } else if (options.LogSqrtScaling) {
    for (auto& [inputTensor, outputTensor] : *dataOpt) {
      if (inputTensor[options.MixedScalingInputVariable].item<TensorDataType>() <= options.MixedScalingThreshold) {
        Utilities::DataProcessor::ScaleLogarithmic(outputTensor);
      } else {
        Utilities::DataProcessor::ScaleSquareRoot(outputTensor);
      }
    }
  }

  if (options.DebugOutput) {
    std::cout << "Get min/max values..." << std::endl;
  }

  // Get min/max values
  bool minMaxInputtedByUser = options.InputMinMaxFilePath != Utilities::DefaultValues::INPUT_MIN_MAX_FILE_PATH;
  if (minMaxInputtedByUser) {
    if (useMixedScaling) {
      auto minMaxFromFile = Utilities::DataProcessor::GetMixedMinMaxFromFile(options.InputMinMaxFilePath,
                                                                             options.NumberOfInputVariables, options.NumberOfOutputVariables);
      if (!minMaxFromFile) {
        return false;
      }
      mixedScalingMinMax = *minMaxFromFile;
    } else {
      auto minMaxFromFile = Utilities::DataProcessor::GetMinMaxFromFile(options.InputMinMaxFilePath,
                                                                        options.NumberOfInputVariables, options.NumberOfOutputVariables);
      if (!minMaxFromFile) {
        return false;
      }
      minMax = *minMaxFromFile;
    }
  } else {
    if (useMixedScaling) {
      Utilities::DataProcessor::CalculateMixedMinMax(*dataOpt, options.MixedScalingInputVariable, options.MixedScalingThreshold, mixedScalingMinMax);
    } else {
      Utilities::DataProcessor::CalculateMinMax(*dataOpt, minMax);
    }
  }

  if (!minMaxValuesAreValid()) {
    if (minMaxInputtedByUser) {
      std::cout << "The inputted min/max values are invalid. A minimum value must not be equal to the corresponding maximum value." << std::endl;
    } else {
      std::cout << "The inputted data is invalid. If no min/max values for the normalization are inputted, each column must contain at least 2 different values." << std::endl;
    }
    return false;
  }

  if (options.DebugOutput) {
    std::cout << "Normalize values..." << std::endl;
  }

  // Normalize
  if (useMixedScaling) {
    for (auto& [inputTensor, outputTensor] : *dataOpt) {
      if (inputTensor[options.MixedScalingInputVariable].item<TensorDataType>() <= options.MixedScalingThreshold) {
        Utilities::DataProcessor::Normalize(outputTensor, mixedScalingMinMax.first.second, -1.0, 0.0); // TODO check if overlap is a problem
      } else {
        Utilities::DataProcessor::Normalize(outputTensor, mixedScalingMinMax.second.second, 0.0, 1.0);
      }
      Utilities::DataProcessor::Normalize(inputTensor, mixedScalingMinMax.first.first, 0.0, 1.0);
    }
  } else {
    Utilities::DataProcessor::Normalize(*dataOpt, minMax, 0.0, 1.0);  // TODO let user control normalization
  }

  // Calculate denormalized mixed scaling threshold value:
  if (useMixedScaling) {
    auto tempInputTensor = dataOpt->front().first.clone();
    tempInputTensor[options.MixedScalingInputVariable] = options.MixedScalingThreshold;

    Utilities::DataProcessor::Normalize(tempInputTensor, mixedScalingMinMax.first.first, 0.0, 1.0);
    normalizedMixedScalingThreshold = tempInputTensor[options.MixedScalingInputVariable].item<TensorDataType>();
  }

  if (options.OutputMinMaxFilePath != Utilities::DefaultValues::OUTPUT_MIN_MAX_FILE_PATH) {
    saveMinMaxToFile();
  }

  if (options.DebugOutput) {
    std::cout << "Configure network..." << std::endl;
  }

  auto networkConfiguration = std::vector<uint32_t>();
  for (uint32_t i = 0; i < options.NumberOfLayers; ++i) {
    networkConfiguration.push_back(options.NumberOfNodesPerLayer);
  }

  network = Network{options.NumberOfInputVariables, options.NumberOfOutputVariables, networkConfiguration};
  analyzer = std::make_unique<NetworkAnalyzer>(network, [this](auto inTensor, auto outTensor, auto limitValues) {
    denormalizeOutputTensor(inTensor, outTensor, limitValues);
  }, [this](auto inTensor, auto outTensor) {
    unscaleOutputTensor(inTensor, outTensor);
  });

  // Load pre-trained weights:
  if (options.InputNetworkParameters != Utilities::DefaultValues::INPUT_NETWORK_PARAMETERS) {
    torch::load(network, options.InputNetworkParameters);
  }

  std::pair<DataVector, DataVector> data;

  if (options.ValidateAfterTraining) {
    data = Utilities::DataSplitter::splitDataRandomly(*dataOpt, 100.0 - options.ValidationPercentage);
  } else {
    data = std::make_pair(*dataOpt, DataVector());
  }

  if (options.BatchVariable.has_value()) {
    useBatchTraining = true;

    batchedTrainingData = Utilities::DataSplitter::splitDataIntoBatches(data.first, options.BatchVariable.value());

    if (options.DebugOutput) {
      std::cout << "Split training data (" << data.first.size() << " data points) into " << batchedTrainingData.size() << " batches." << std::endl;
      std::cout << "Size of first batch: " << batchedTrainingData.begin()->second.size() << std::endl;
    }
  }

  if (options.DebugOutput) {
    std::cout << "Start the training..." << std::endl;
  }

  trainNetwork(data.first);

  if (options.DebugOutput) {
    std::cout << "\nTraining finished." << std::endl;
  }

  network->eval();

  if (options.OutputNetworkParameters != Utilities::DefaultValues::OUTPUT_NETWORK_PARAMETERS) {
    torch::save(network, options.OutputNetworkParameters);
  }

  if (options.OutputValuesFilePath != Utilities::DefaultValues::OUTPUT_VALUE) {
    saveValuesToFile(*dataOpt, options.OutputValuesFilePath);
  }

  if (options.OutputDiffFilePath != Utilities::DefaultValues::OUTPUT_DIFF) {
    saveDiffToFile(*dataOpt, options.OutputDiffFilePath, false);
  }

  if (options.OutputRelativeDiffFilePath != Utilities::DefaultValues::OUTPUT_RELATIVE_DIFF) {
    saveDiffToFile(*dataOpt, options.OutputRelativeDiffFilePath, true);
  }

  if (options.SaveProgressFilePath != Utilities::DefaultValues::PROGRESS_FILE_PATH) {
    Utilities::FileParser::SaveProgressData(trainingProgress, options.SaveProgressFilePath);
  }

  // Output behaviour of network:
  if (options.PrintBehaviour) {
    std::cout << std::endl;
    if (options.ValidateAfterTraining) {
      std::cout << "R2 score (training): " << analyzer->calculateR2Score(data.first) << std::endl;
      std::cout << "R2 score alternate (training): " << analyzer->calculateR2ScoreAlternate(data.first) << std::endl;
      std::cout << "R2 score alternate denormalized (training): " << analyzer->calculateR2ScoreAlternateDenormalized(data.first) << std::endl;
      std::cout << "R2 score (validation): " << analyzer->calculateR2Score(data.second) << std::endl;
      std::cout << "R2 score alternate (validation): " << analyzer->calculateR2ScoreAlternate(data.second) << std::endl;
      std::cout << "R2 score alternate denormalized (validation): " << analyzer->calculateR2ScoreAlternateDenormalized(data.second) << std::endl;
    }
    std::cout << "R2 score (all): " << analyzer->calculateR2Score(*dataOpt) << std::endl;
    std::cout << "R2 score alternate (all): " << analyzer->calculateR2ScoreAlternate(*dataOpt) << std::endl;
    std::cout << "R2 score alternate denormalized (all): " << analyzer->calculateR2ScoreAlternateDenormalized(*dataOpt) << std::endl;

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

void Logic::trainNetwork(DataVector const& data)
{
  if (data.empty()) {
    return;
  }

  auto const& numberOfEpochs = options.NumberOfEpochs;
  bool saveProgress = options.SaveProgressFilePath != Utilities::DefaultValues::PROGRESS_FILE_PATH;

  torch::optim::SGD optimizer(network->parameters(), options.LearnRate);

  auto lastMeanError = analyzer->calculateMeanSquaredError(data);
  auto currentMeanError = lastMeanError;

  bool continueTraining = true;
  uint32_t numberOfDeteriorationsInRow = 0;

  auto start = std::chrono::steady_clock::now();

  for (uint32_t epoch = 1; epoch <= numberOfEpochs || continueTraining; ++epoch) {
    auto elapsed = std::chrono::duration_cast<TimeoutDuration>(std::chrono::steady_clock::now() - start);
    auto remaining = ((elapsed / std::max(epoch - 1, 1u)) * (numberOfEpochs - epoch + 1));
    lastMeanError = currentMeanError;
    currentMeanError = analyzer->calculateMeanSquaredError(data);

    if (lastMeanError - currentMeanError < options.Epsilon) {
      ++numberOfDeteriorationsInRow;
      if (numberOfDeteriorationsInRow > options.NumberOfDeteriorations) {
        continueTraining = false;
      }
    } else {
      numberOfDeteriorationsInRow = 0;
    }

    if (saveProgress) {
      auto r2score = analyzer->calculateR2ScoreAlternate(data);
      trainingProgress.emplace_back(LearnProgressDataSet{
        epoch,
        r2score,
        currentMeanError,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count())
      });
    }

    if (options.ShowProgressDuringTraining) {
      if (epoch > numberOfEpochs) {
        std::cout << "\rContinue training. Mean squared error changed from " << lastMeanError << " to " << currentMeanError << " -- epoch: " << epoch;
        std::flush(std::cout);
      } else {
        std::cout << "\rEpoch " << epoch << " of " << numberOfEpochs << ". Current mean squared error: " << currentMeanError << " previous: " << lastMeanError <<
                  " -- Remaining time: " << formatDuration<std::chrono::milliseconds, std::chrono::hours, std::chrono::minutes, std::chrono::seconds>(remaining); // TODO better output
        std::flush(std::cout);
      }
    }

    if (elapsed > options.MaxExecutionTime) {
      std::cout << "\nStop execution (timeout)." << std::endl;
      break;
    }

    if (std::isnan(currentMeanError)) {
      std::cout << "\nStop execution (error is NaN)." << std::endl;
      break;
    }

    if (useBatchTraining) {
      for (auto const& [identifier, batch] : batchedTrainingData) {
        (void) identifier;

        optimizer.zero_grad();

        for (auto const& [x, y] : batch) {
          auto prediction = network->forward(x);
          auto loss = torch::mse_loss(prediction, y);

          loss.backward();
        }

        optimizer.step();
      }
    } else {
      for (auto const& [x, y] : data) {
        auto prediction = network->forward(x);

        auto loss = torch::mse_loss(prediction, y);

        optimizer.zero_grad();

        loss.backward();
        optimizer.step();
      }
    }
  }

  if (options.DebugOutput) {
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
      Utilities::DataProcessor::Normalize(inTensor, (useMixedScaling) ? mixedScalingMinMax.first.first : inputMinMax, 0.0, 1.0);
      auto output = network->forward(inTensor);
      auto dOutputTensor = output.clone();
      denormalizeOutputTensor(inTensor, dOutputTensor, false);

      unscaleOutputTensor(inTensor, dOutputTensor);

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

void Logic::outputBehaviour(DataVector const& data)
{
  for (auto const& [inputTensor, outputTensor] : data) {
    auto prediction = network->forward(inputTensor);
    auto loss = torch::mse_loss(prediction, outputTensor);

    torch::Tensor dInputTensor = inputTensor.clone();
    torch::Tensor dOutputTensor = outputTensor.clone();
    torch::Tensor dPrediction = prediction.clone();

    denormalizeInputTensor(dInputTensor, false);
    denormalizeOutputTensor(inputTensor, dOutputTensor, false);
    denormalizeOutputTensor(inputTensor, dPrediction, false);

    unscaleOutputTensor(inputTensor, dOutputTensor);
    unscaleOutputTensor(inputTensor, dPrediction);

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

    denormalizeInputTensor(dInputTensor, false);
    denormalizeOutputTensor(inputTensor, prediction, false);

    unscaleOutputTensor(inputTensor, prediction);

    values[i++] = std::make_pair(dInputTensor, prediction);
  }

  Utilities::FileParser::SaveData(values, path, inputFileHeader);
}

void Logic::saveDiffToFile(DataVector const& data, std::string const& path, bool outputRelativeDiff)
{
  DataVector diff(data.size());

  size_t i = 0;
  for (auto const& [inputTensor, outputTensor] : data) {
    auto prediction = network->forward(inputTensor);
    torch::Tensor dInputTensor = inputTensor.clone();
    torch::Tensor dOutputTensor = outputTensor.clone();

    denormalizeInputTensor(dInputTensor, false);
    denormalizeOutputTensor(inputTensor, dOutputTensor, false);
    denormalizeOutputTensor(inputTensor, prediction, false);

    unscaleOutputTensor(inputTensor, dOutputTensor);
    unscaleOutputTensor(inputTensor, prediction);

    torch::Tensor difference;
    if (outputRelativeDiff) {
      difference = NetworkAnalyzer::calculateRelativeDiff(dOutputTensor, prediction);
    } else {
      difference = NetworkAnalyzer::calculateDiff(dOutputTensor, prediction);
    }

    diff[i++] = std::make_pair(dInputTensor, difference);
  }

  Utilities::FileParser::SaveData(diff, path, inputFileHeader);
}

void Logic::saveMinMaxToFile() const
{
  auto inTensorDefault = torch::zeros(options.NumberOfInputVariables, TORCH_DATA_TYPE);
  auto outTensorDefault = torch::zeros(options.NumberOfOutputVariables, TORCH_DATA_TYPE);

  DataVector data = (useMixedScaling) ? DataVector(4) : DataVector(2);
  data[0] = std::make_pair(inTensorDefault, outTensorDefault);
  data[1] = std::make_pair(inTensorDefault.clone(), outTensorDefault.clone());
  if (useMixedScaling) {
    data[2] = std::make_pair(inTensorDefault.clone(), outTensorDefault.clone());
    data[3] = std::make_pair(inTensorDefault.clone(), outTensorDefault.clone());
  }

  for (uint32_t i = 0; i < options.NumberOfInputVariables; ++i) {
    if (useMixedScaling) {
      data[0].first[i] = mixedScalingMinMax.first.first[i].first;
      data[1].first[i] = mixedScalingMinMax.first.first[i].second;
      data[2].first[i] = mixedScalingMinMax.second.first[i].first;
      data[3].first[i] = mixedScalingMinMax.second.first[i].second;
    } else {
      data[0].first[i] = inputMinMax[i].first;
      data[1].first[i] = inputMinMax[i].second;
    }
  }

  for (uint32_t i = 0; i < options.NumberOfOutputVariables; ++i) {
    if (useMixedScaling) {
      data[0].second[i] = mixedScalingMinMax.first.second[i].first;
      data[1].second[i] = mixedScalingMinMax.first.second[i].second;
      data[2].second[i] = mixedScalingMinMax.second.second[i].first;
      data[3].second[i] = mixedScalingMinMax.second.second[i].second;
    } else {
      data[0].second[i] = outputMinMax[i].first;
      data[1].second[i] = outputMinMax[i].second;
    }
  }

  Utilities::FileParser::SaveData(data, options.OutputMinMaxFilePath, inputFileHeader);
}

inline void Logic::denormalizeInputTensor(torch::Tensor& tensor, bool limitValues)
{
  Utilities::DataProcessor::Denormalize(tensor, (useMixedScaling) ? mixedScalingMinMax.first.first : inputMinMax, 0.0, 1.0, limitValues);
}

inline void Logic::denormalizeOutputTensor(torch::Tensor const& inputTensor, torch::Tensor& outputTensor, bool limitValues)
{
  if (useMixedScaling) {
    if (inputTensor[options.MixedScalingInputVariable].item<TensorDataType>() <= normalizedMixedScalingThreshold) {
      Utilities::DataProcessor::Denormalize(outputTensor, mixedScalingMinMax.first.second, -1.0, 0.0, limitValues);
    } else {
      Utilities::DataProcessor::Denormalize(outputTensor, mixedScalingMinMax.second.second, 0.0, 1.0, limitValues);
    }
  } else {
    Utilities::DataProcessor::Denormalize(outputTensor, outputMinMax, 0.0, 1.0, limitValues);
  }
}

void Logic::unscaleOutputTensor(torch::Tensor const& inputTensor, torch::Tensor& outputTensor) const
{
  if (options.LogScaling) {
    Utilities::DataProcessor::UnscaleLogarithmic(outputTensor);
  }
  else if (options.SqrtScaling) {
    Utilities::DataProcessor::UnscaleSquareRoot(outputTensor);
  }
  else if (options.LogLinScaling) {
    if (inputTensor[options.MixedScalingInputVariable].item<TensorDataType>() <= normalizedMixedScalingThreshold) {
      Utilities::DataProcessor::UnscaleLogarithmic(outputTensor);
    }
  }
  else if (options.LogSqrtScaling) {
    if (inputTensor[options.MixedScalingInputVariable].item<TensorDataType>() <= normalizedMixedScalingThreshold) {
      Utilities::DataProcessor::UnscaleLogarithmic(outputTensor);
    } else {
      Utilities::DataProcessor::UnscaleSquareRoot(outputTensor);
    }
  }
}

bool Logic::minMaxValuesAreValid() const
{
  auto validationFunction = [] (std::vector<std::pair<TensorDataType, TensorDataType>> const& data) {
    for (auto const& [min, max] : data) {
      if (min == max) {
        return false;
      }
    }
    return true;
  };

  if (useMixedScaling) {
    if (!validationFunction(mixedScalingMinMax.first.first) || !validationFunction(mixedScalingMinMax.first.second) ||
        !validationFunction(mixedScalingMinMax.second.first) || !validationFunction(mixedScalingMinMax.second.second)) {
      return false;
    }
  } else {
    if (!validationFunction(inputMinMax) || !validationFunction(outputMinMax)) {
      return false;
    }
  }

  return true;
}

}
