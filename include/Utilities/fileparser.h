#pragma once

#include "Utilities/constants.h"

#include <optional>

namespace Utilities {

class FileParser
{
public:
  static std::optional<DataVector> ParseInputFile(const std::string& path, uint32_t numberOfInputNodes, uint32_t numberOfOutputNodes);
};

}
