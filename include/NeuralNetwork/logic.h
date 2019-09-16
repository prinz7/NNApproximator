#pragma once

#include "Utilities/programoptions.h"

namespace NeuralNetwork {

class Logic
{
public:
  [[nodiscard]]
  bool performUserRequest(const Utilities::ProgramOptions& options);
};

}
