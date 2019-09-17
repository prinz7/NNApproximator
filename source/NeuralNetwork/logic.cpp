#include "NeuralNetwork/logic.h"

#include "Utilities/fileparser.h"

namespace NeuralNetwork {

bool Logic::performUserRequest(const Utilities::ProgramOptions &options)
{
  auto data = Utilities::FileParser::ParseInputFile(options.InputFilePath, options.NumberOfInputVariables, options.NumberOfOutputVariables);
  if (!data) {
    return false;
  }

  return true;
}

}
