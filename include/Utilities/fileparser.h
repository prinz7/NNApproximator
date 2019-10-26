#pragma once

#include "Utilities/constants.h"

#include <optional>

namespace Utilities {

class FileParser
{
public:
  static std::optional<DataVector> ParseInputFile(std::string const& path, uint32_t numberOfInputNodes, uint32_t numberOfOutputNodes, std::string& fileHeader);
  static void SaveData(DataVector const& data, std::string const& outputFilePath, std::string const& fileHeader);
  static void SaveProgressData(ProgressVector const& data, std::string const& filePath);
};

}
