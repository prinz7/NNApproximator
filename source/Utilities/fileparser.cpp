#include "Utilities/fileparser.h"

#include <iostream>

namespace Utilities {

std::optional<DataVector> FileParser::ParseInputFile(const std::string& path, uint32_t numberOfInputNodes, uint32_t numberOfOutputNodes)
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

  // TODO: Use Header of input file

  while (std::getline(inputFile, line)) {
    std::istringstream iss(line);
    TensorDataType value;

    // get input:
    auto inTensor = torch::zeros(numberOfInputNodes, torch::kFloat);
    for (uint32_t i = 0; i < numberOfInputNodes; ++i) {
      if (!(iss >> value)) {
        std::cout << "Error: Unable to parse input data." << std::endl;
        return std::nullopt;
      }
      inTensor[i] = value;
    }

    // get output:
    auto outTensor = torch::zeros(numberOfOutputNodes, torch::kFloat);
    for (uint32_t i = 0; i < numberOfOutputNodes; ++i) {
      if (!(iss >> value)) {
        std::cout << "Error: Unable to parse output data." << std::endl;
        return std::nullopt;
      }
      outTensor[i] = value;
    }

    data.emplace_back(std::make_pair(inTensor, outTensor));
  }

  return std::make_optional(data);
}

}
