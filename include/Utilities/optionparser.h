#pragma once

#include "Utilities/programoptions.h"

namespace Utilities {

class OptionParser
{
public:
  [[nodiscard]]
  static std::optional<ProgramOptions> ParseCommandLineParameters(int argc, char* argv[]);
};

}
