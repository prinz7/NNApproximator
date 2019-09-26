#pragma once

#include <map>
#include <string>

namespace Utilities {

using FilePath = std::string;

namespace DefaultValues {

const FilePath INPUT_DATA_FILE_PATH = {};
const FilePath INPUT_NETWORK_PARAMETERS = {};
const FilePath OUTPUT_NETWORK_PARAMETERS = {};
const uint32_t NUMBER_OF_INPUT_VARIABLES = 1;
const uint32_t NUMBER_OF_OUTPUT_VARIABLES = 1;
const uint32_t NUMBER_OF_EPOCHS = 10;
const bool     SHOW_PROGRESS_DURING_TRAINING = true;
const bool     INTERACTIVE_MODE = false;
const double   EPSILON = 1.0;

const std::string CLI_HELP_TEXT = {
  std::string("List of possible commandline parameters:\n") +
  "--help | -h                        : Output this text message.\n" +
  "--input <filepath> | -i <filepath> : Use content of <filepath> for the input data.\n" +
  "--numberIn X | -ni X               : Sets the number of input variables to X. Default: " + std::to_string(NUMBER_OF_INPUT_VARIABLES) + "\n" +
  "--numberOut X | -no X              : Sets the number of output variables to X. Default: " + std::to_string(NUMBER_OF_OUTPUT_VARIABLES) + "\n" +
  "--epochs X | -e X                  : Sets the number of epochs (how many times the data is used for training). Default: " + std::to_string(NUMBER_OF_EPOCHS) + "\n" +
  "--showProgress <bool>              : Activate or deactivate display of progress and eta of the training. Default: " + (SHOW_PROGRESS_DURING_TRAINING ? "true" : "false") + "\n" +
  "--inWeights <filepath>             : If set loads the weights in the file for the network in the initialization phase.\n" +
  "--outWeights <filepath>            : If set saves the weights of the network to the specified file after the training phase.\n" +
  "--interactive                      : If set activated the interactive mode after the training to test user input on the neural network.\n" +
  "--epsilon <double>                 : If set continues training after the last epoch until the improvement of the mean squarred error is less than the set epsilon. Default: " + std::to_string(EPSILON) + "\n"
};

}

enum class CLIParameters
{
  Help, InputFilePath, NumberOfInputVariabes, NumberOfOutputVariables, NumberOfEpochs, ShowProgressDuringTraining, InputNetworkParameters,
  OutputNetworkParameters, Interactive, Epsilon
};

const std::map<std::string, CLIParameters> CLIParameterMap {
  {"--help",         CLIParameters::Help},
  {"-h",             CLIParameters::Help},
  {"--input",        CLIParameters::InputFilePath},
  {"-i",             CLIParameters::InputFilePath},
  {"--numberIn",     CLIParameters::NumberOfInputVariabes},
  {"-ni",            CLIParameters::NumberOfInputVariabes},
  {"--numberOut",    CLIParameters::NumberOfOutputVariables},
  {"-no",            CLIParameters::NumberOfOutputVariables},
  {"--epochs",       CLIParameters::NumberOfEpochs},
  {"-e",             CLIParameters::NumberOfEpochs},
  {"--showProgress", CLIParameters::ShowProgressDuringTraining},
  {"--inWeights",    CLIParameters::InputNetworkParameters},
  {"--outWeights",   CLIParameters::OutputNetworkParameters},
  {"--interactive",  CLIParameters::Interactive},
  {"--epsilon",      CLIParameters::Epsilon}
};

class ProgramOptions
{
public:
  FilePath InputDataFilePath {          DefaultValues::INPUT_DATA_FILE_PATH };
  FilePath InputNetworkParameters {     DefaultValues::INPUT_NETWORK_PARAMETERS };
  FilePath OutputNetworkParameters {    DefaultValues::OUTPUT_NETWORK_PARAMETERS };
  uint32_t NumberOfInputVariables {     DefaultValues::NUMBER_OF_INPUT_VARIABLES };
  uint32_t NumberOfOutputVariables {    DefaultValues::NUMBER_OF_OUTPUT_VARIABLES };
  uint32_t NumberOfEpochs {             DefaultValues::NUMBER_OF_EPOCHS };
  bool     ShowProgressDuringTraining { DefaultValues::SHOW_PROGRESS_DURING_TRAINING };
  bool     InteractiveMode {            DefaultValues::INTERACTIVE_MODE };
  double   Epsilon {                    DefaultValues::EPSILON };
};

}
