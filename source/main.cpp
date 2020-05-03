#include "NeuralNetwork/logic.h"
#include "Utilities/optionparser.h"

int main(int argc, char* argv[]) {
  auto options = Utilities::OptionParser::ParseCommandLineParameters(argc, argv);
  if (options == std::nullopt) {
    return 1;
  }

  NeuralNetwork::Logic logic{};
  if (!logic.performUserRequest(*options)) {
    return 2;
  }

  return 0;
}
