#include "Utilities/dataprocessor.h"
#include "Utilities/datasplitter.h"
#include "Utilities/fileparser.h"

namespace Utilities {

namespace {

const TensorDataType MINIMUM_ALLOWED_VALUE = 1e-30;

}

void DataProcessor::CalculateMinMax(DataVector const& data, MinMaxValues& minMaxVectors)
{
  if (data.empty()) return;
  auto& inputMinMax = minMaxVectors.first;
  auto& outputMinMax = minMaxVectors.second;
  auto numberOfInputNodes = data.front().first.size(0);
  auto numberOfOutputNodes = data.front().second.size(0);

  // Initialize minMaxVectors:
  inputMinMax = MinMaxVector(numberOfInputNodes, std::make_pair(std::numeric_limits<TensorDataType>::max(), std::numeric_limits<TensorDataType>::lowest()));
  outputMinMax = MinMaxVector(numberOfOutputNodes, std::make_pair(std::numeric_limits<TensorDataType>::max(), std::numeric_limits<TensorDataType>::lowest()));

  // Calculate min and max:
  for (auto& [inputTensor, outputTensor] : data) {
    for (int64_t i = 0; i < numberOfInputNodes; ++i) {
      inputMinMax[i].first = std::min(inputMinMax[i].first, inputTensor[i].item<TensorDataType>());
      inputMinMax[i].second = std::max(inputMinMax[i].second, inputTensor[i].item<TensorDataType>());
    }

    for (int64_t i = 0; i < numberOfOutputNodes; ++i) {
      outputMinMax[i].first = std::min(outputMinMax[i].first, outputTensor[i].item<TensorDataType>());
      outputMinMax[i].second = std::max(outputMinMax[i].second, outputTensor[i].item<TensorDataType>());
    }
  }
}

void DataProcessor::CalculateMixedMinMax(DataVector const& data, uint32_t thresholdVariable, TensorDataType threshold, MixedMinMaxValues& mixedMinMaxValues)
{
  mixedMinMaxValues = std::make_pair(MinMaxValues(), MinMaxValues());

  auto splitData = DataSplitter::splitDataWithThreshold(data, thresholdVariable, threshold);

  CalculateMinMax(splitData.first, mixedMinMaxValues.first);
  CalculateMinMax(splitData.second, mixedMinMaxValues.second);

  // Only split min/max values for the output tensor:
  MinMaxValues globalMinMax{};

  CalculateMinMax(data, globalMinMax);
  mixedMinMaxValues.first.first = globalMinMax.first;
  mixedMinMaxValues.second.first = globalMinMax.first;
}

std::optional<MinMaxValues> DataProcessor::GetMinMaxFromFile(FilePath const& filePath, uint32_t numberOfInputVariables, uint32_t numberOfOutputVariables)
{
  std::string fileHeader{};
  auto minMaxOpt = Utilities::FileParser::ParseInputFile(filePath, numberOfInputVariables, numberOfOutputVariables, fileHeader);
  if (!minMaxOpt) {
    return std::nullopt;
  }

  if (minMaxOpt->size() != 2) {
    std::cout << "Error: File with min/max values has the wrong number of data. Expected 2 values for each column, got: " + std::to_string(minMaxOpt->size()) << std::endl;
    return std::nullopt;
  }

  std::pair<MinMaxVector, MinMaxVector> minMaxValues{
    MinMaxVector(numberOfInputVariables),
    MinMaxVector(numberOfOutputVariables)
  };

  auto& inputMinMax = minMaxValues.first;
  auto& outputMinMax = minMaxValues.second;
  auto const& fileMinMax = *minMaxOpt;

  for (uint32_t j = 0; j < numberOfInputVariables; ++j) {
    inputMinMax[j].first = fileMinMax[0].first[j].item<TensorDataType>();
    inputMinMax[j].second = fileMinMax[1].first[j].item<TensorDataType>();
  }
  for (uint32_t j = 0; j < numberOfOutputVariables; ++j) {
    outputMinMax[j].first = fileMinMax[0].second[j].item<TensorDataType>();
    outputMinMax[j].second = fileMinMax[1].second[j].item<TensorDataType>();
  }

  return std::make_optional(minMaxValues);
}

std::optional<MixedMinMaxValues> DataProcessor::GetMixedMinMaxFromFile(FilePath const& filePath, uint32_t numberOfInputVariables, uint32_t numberOfOutputVariables)
{
  std::string fileHeader{};
  auto minMaxOpt = Utilities::FileParser::ParseInputFile(filePath, numberOfInputVariables, numberOfOutputVariables, fileHeader);
  if (!minMaxOpt) {
    return std::nullopt;
  }

  if (minMaxOpt->size() != 4) {
    std::cout << "Error: File with [mixed scaling] min/max values has the wrong number of data. Expected 4 values for each column, got: " + std::to_string(minMaxOpt->size()) << std::endl;
    return std::nullopt;
  }

  std::pair<MinMaxVector, MinMaxVector> minMaxValues1{
    MinMaxVector(numberOfInputVariables),
    MinMaxVector(numberOfOutputVariables)
  };
  std::pair<MinMaxVector, MinMaxVector> minMaxValues2{
    MinMaxVector(numberOfInputVariables),
    MinMaxVector(numberOfOutputVariables)
  };

  auto& inputMinMax1 = minMaxValues1.first;
  auto& outputMinMax1 = minMaxValues1.second;
  auto& inputMinMax2 = minMaxValues2.first;
  auto& outputMinMax2 = minMaxValues2.second;
  auto const& fileMinMax = *minMaxOpt;

  for (uint32_t j = 0; j < numberOfInputVariables; ++j) {
    inputMinMax1[j].first = fileMinMax[0].first[j].item<TensorDataType>();
    inputMinMax1[j].second = fileMinMax[1].first[j].item<TensorDataType>();
    inputMinMax2[j].first = fileMinMax[2].first[j].item<TensorDataType>();
    inputMinMax2[j].second = fileMinMax[3].first[j].item<TensorDataType>();
  }
  for (uint32_t j = 0; j < numberOfOutputVariables; ++j) {
    outputMinMax1[j].first = fileMinMax[0].second[j].item<TensorDataType>();
    outputMinMax1[j].second = fileMinMax[1].second[j].item<TensorDataType>();
    outputMinMax2[j].first = fileMinMax[2].second[j].item<TensorDataType>();
    outputMinMax2[j].second = fileMinMax[3].second[j].item<TensorDataType>();
  }

  return std::make_optional(std::make_pair(minMaxValues1, minMaxValues2));
}

void DataProcessor::Normalize(DataVector& data, std::pair<MinMaxVector const, MinMaxVector const> const& minMaxVectors, TensorDataType const newMinValue, TensorDataType const newMaxValue)
{
  auto const& inputMinMax = minMaxVectors.first;
  auto const& outputMinMax = minMaxVectors.second;

  for (auto& [inputTensor, outputTensor] : data) {
    Normalize(inputTensor, inputMinMax, newMinValue, newMaxValue);
    Normalize(outputTensor, outputMinMax, newMinValue, newMaxValue);
  }
}

void DataProcessor::Normalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType const newMinValue, TensorDataType const newMaxValue)
{
  // Normalize data -- use '(X - min) / (max - min)' to get values between 0 and 1:
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    TensorDataType normalizationFactor = newMaxValue - newMinValue;
    tensor[i] = ((tensor[i].item<TensorDataType>() - minMaxVector[i].first) / (minMaxVector[i].second - minMaxVector[i].first)) * normalizationFactor + newMinValue;
  }
}

void DataProcessor::Denormalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType const oldMinValue, TensorDataType const oldMaxValue, bool const limitValues)
{
  if (tensor.size(0) != static_cast<int64_t>(minMaxVector.size())) {
    return;
  }

  TensorDataType normalizationFactor = oldMaxValue - oldMinValue;
  for (size_t i = 0; i < minMaxVector.size(); ++i) {
    if (limitValues) {
      tensor[i] = std::max(oldMinValue, (std::min(oldMaxValue, tensor[i].item<TensorDataType>())));
    }
    tensor[i] = (((tensor[i] - oldMinValue) / normalizationFactor) * (minMaxVector[i].second - minMaxVector[i].first)) + minMaxVector[i].first;
  }
}

void DataProcessor::ScaleLogarithmic(torch::Tensor& data)
{
  for (int64_t i = 0; i < data.size(0); ++i) {
    if (data[i].item<TensorDataType>() < MINIMUM_ALLOWED_VALUE) {
      data[i] = MINIMUM_ALLOWED_VALUE;
    }
    data[i] = std::log(data[i].item<TensorDataType>());
  }
}

void DataProcessor::UnscaleLogarithmic(torch::Tensor& data)
{
  for (int64_t i = 0; i < data.size(0); ++i) {
    data[i] = std::exp(data[i].item<TensorDataType>());
  }
}

void DataProcessor::ScaleSquareRoot(torch::Tensor& data)
{
  for (int64_t i = 0; i < data.size(0); ++i) {
    if (data[i].item<TensorDataType>() < MINIMUM_ALLOWED_VALUE) {
      data[i] = MINIMUM_ALLOWED_VALUE;
    }
    data[i] = std::sqrt(data[i].item<TensorDataType>());
  }
}

void DataProcessor::UnscaleSquareRoot(torch::Tensor& data)
{
  for (int64_t i = 0; i < data.size(0); ++i) {
    data[i] = std::pow(data[i].item<TensorDataType>(), 2.0);
  }
}

}
