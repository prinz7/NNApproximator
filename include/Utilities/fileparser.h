#pragma once

#include "Utilities/constants.h"

#include <optional>

namespace Utilities {

class FileParser
{
public:
  /*
   * Parses the given file and returns the data and the file header.
   */
  static std::optional<DataVector> ParseInputFile(std::string const& path, uint32_t numberOfInputNodes, uint32_t numberOfOutputNodes, std::string& fileHeader);
  /*
   * Saves the data to given file path together with the given file header.
   */
  static void SaveData(DataVector const& data, std::string const& outputFilePath, std::string const& fileHeader);
  /*
   * Saves the given progress data to the given file path.
   */
  static void SaveProgressData(ProgressVector const& data, std::string const& filePath);
};

}
