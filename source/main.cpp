#include "Utilities/optionparser.h"
#include "MachineLearningTest/torchtest.h"

int main(int argc, char* argv[]) {
  auto options = Utilities::OptionParser::ParseCommandLineParameters(argc, argv);
  TorchTest::TorchTest helper;

  helper.run();

  return 0;
}
