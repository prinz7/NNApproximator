#pragma once

#include "Utilities/programoptions.h"

namespace Utilities {

class OptionParser
{
public:
  /*
   * Parses the command line parameters and performs some sanity checks on it.
   */
  [[nodiscard]]
  static std::optional<ProgramOptions> ParseCommandLineParameters(int argc, char* argv[]);
};

}
