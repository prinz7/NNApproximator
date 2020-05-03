#include "Utilities/fileparser.h"

#include <iostream>

namespace Utilities {

std::optional<DataVector> FileParser::ParseInputFile(std::string const& path, uint32_t const numberOfInputNodes, uint32_t const numberOfOutputNodes, std::string& fileHeader)
{
  auto data = DataVector();

  if (path.empty()) {
    std::cout << "Error: \"" << path << "\" is not a valid path to a file for the input data." << std::endl;
    return std::nullopt;
  }

  std::ifstream inputFile(path);

  std::string line;
  if (!std::getline(inputFile, line)) {
    std::cout << "Error: Inputfile is empty or not valid." << std::endl;
    return std::nullopt;
  }
  fileHeader = line;

  while (std::getline(inputFile, line)) {
    line.erase (std::remove(line.begin(), line.end(), ','), line.end());  // remove ',' from string
    std::istringstream iss(line);
    TensorDataType value;

    // get input:
    auto inTensor = torch::zeros(numberOfInputNodes, TORCH_DATA_TYPE);
    for (uint32_t i = 0; i < numberOfInputNodes; ++i) {
      if (!(iss >> value)) {
        std::cout << "Error: Unable to parse input data." << std::endl;
        return std::nullopt;
      }
      inTensor[i] = value;
    }

    // get output:
    auto outTensor = torch::zeros(numberOfOutputNodes, TORCH_DATA_TYPE);
    for (uint32_t i = 0; i < numberOfOutputNodes; ++i) {
      if (!(iss >> value)) {
        std::cout << "Error: Unable to parse output data." << std::endl;
        return std::nullopt;
      }
      outTensor[i] = value;
    }

    data.emplace_back(std::make_pair(inTensor, outTensor));
  }

  inputFile.close();
  return std::make_optional(data);
}

void FileParser::SaveData(DataVector const& data, std::string const& outputFilePath, std::string const& fileHeader)
{
  if (data.empty()) {
    return;
  }
  std::ofstream outputFile(outputFilePath);

  outputFile << fileHeader << "\n";

  for (auto const& [inputTensor, outputTensor] : data) {
    outputFile << inputTensor[0].item<TensorDataType>();

    for (int64_t i = 1; i < inputTensor.size(0); ++i) {
      outputFile << ", " << inputTensor[i].item<TensorDataType>();
    }

    for (int64_t i = 0; i < outputTensor.size(0); ++i) {
      outputFile << ", " << outputTensor[i].item<TensorDataType>();
    }

    outputFile << "\n";
  }

  outputFile.close();
}

void FileParser::SaveProgressData(ProgressVector const& data, std::string const& filePath)
{
  if (data.empty()) {
    return;
  }
  std::ofstream outputFile(filePath);

  outputFile << LEARN_PROGRESS_FILE_HEADER_FIRST_PART;
  for (size_t i = 1; i <= data[0].r2Score.size(); ++i) {
    outputFile << LEARN_PROGRESS_R2_SCORE_HEADER_PART << i;
  }
  outputFile << "\n";

  for (auto const& [epoch, r2score, meanSquaredError, elapsedTimeInMS] : data) {
    outputFile << epoch << ", " << meanSquaredError << ", " << elapsedTimeInMS;
    for (auto score : r2score) {
      outputFile << ", " << score;
    }
    outputFile << "\n";
  }

  outputFile.close();
}

}
