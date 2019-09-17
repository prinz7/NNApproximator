#pragma once

#include <map>
#include <string>

namespace Utilities {

using FilePath = std::string;

namespace DefaultValues {

const FilePath INPUT_FILE_PATH = {};
const uint32_t NUMBER_OF_INPUT_VARIABLES = 1;
const uint32_t NUMBER_OF_OUTPUT_VARIABLES = 1;
const bool     SHOW_PROGRESS_DURING_TRAINING = true;

const std::string CLI_HELP_TEXT = {
  std::string("List of possible commandline parameters:\n") +
  "--help | -h                        : Output this text message.\n" +
  "--input <filepath> | -i <filepath> : Use content of <filepath> for the input data.\n" +
  "--numberIn X | -ni X               : Sets the number of input variables to X. Default: " + std::to_string(NUMBER_OF_INPUT_VARIABLES) + "\n" +
  "--numberOut X | -no X              : Sets the number of output variables to X. Default: " + std::to_string(NUMBER_OF_OUTPUT_VARIABLES) + "\n" +
  "--showProgress <bool>              : Activate or deactivate display of progress and eta of the training. Default: " + (SHOW_PROGRESS_DURING_TRAINING ? "true" : "false") + "\n"
};

}

enum class CLIParameters
{
  Help, InputFilePath, NumberOfInputVariabes, NumberOfOutputVariables, ShowProgressDuringTraining
};

const std::map<std::string, CLIParameters> CLIParameterMap {
  {"--help",          CLIParameters::Help},
  {"-h",              CLIParameters::Help},
  {"--input",         CLIParameters::InputFilePath},
  {"-i",              CLIParameters::InputFilePath},
  {"--numberIn",      CLIParameters::NumberOfInputVariabes},
  {"-ni",             CLIParameters::NumberOfInputVariabes},
  {"--numberOut",     CLIParameters::NumberOfOutputVariables},
  {"-no",             CLIParameters::NumberOfOutputVariables},
  {"--showProgress",  CLIParameters::ShowProgressDuringTraining}
};

class ProgramOptions
{
public:
  FilePath InputFilePath {              DefaultValues::INPUT_FILE_PATH };
  uint32_t NumberOfInputVariables {     DefaultValues::NUMBER_OF_INPUT_VARIABLES };
  uint32_t NumberOfOutputVariables {    DefaultValues::NUMBER_OF_OUTPUT_VARIABLES };
  bool     ShowProgressDuringTraining { DefaultValues::SHOW_PROGRESS_DURING_TRAINING };
};

}
