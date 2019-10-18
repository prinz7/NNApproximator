#pragma once

#include <torch/torch.h>

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
const bool     VALIDATE_AFTER_TRAINING = false;
const double   VALIDATION_PERCENTAGE = 30.0;
const FilePath OUTPUT_VALUE = {};
const FilePath OUTPUT_DIFF = {};
const bool     PRINT_BEHAVIOUR = false;
const int32_t  NUMBER_OF_THREADS = torch::get_num_threads();

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
  "--epsilon <double>                 : If set continues training after the last epoch until the improvement of the mean squarred error is less than the set epsilon. Default: " + std::to_string(EPSILON) + "\n" +
  "--validate                         : If set splits the data set in a training and validation set. After the training the network is tested with the validation set.\n" +
  "--validatePercentage <double>      : Sets the percentage of the data, which is only used for validation and not for training. Value should be between 0 and 100. Default: " + std::to_string(VALIDATION_PERCENTAGE) + "\n" +
  "--outValues <filepath>             : If set saves the output of the neural network for all input values to the specified file.\n" +
  "--outDiff <filepath>               : If set saves the difference of the output of the neural network and given input values to the specified file.\n" +
  "--printBehaviour                   : If set outputs the behaviour of the neural network to the console for the given input values.\n" +
  "--threads X | -t X                 : Sets the number of used threads to X. Default value depends on the given system. Default value of the current system: " + std::to_string(NUMBER_OF_THREADS) + "\n"
};

}

enum class CLIParameters
{
  Help, InputFilePath, NumberOfInputVariables, NumberOfOutputVariables, NumberOfEpochs, ShowProgressDuringTraining, InputNetworkParameters,
  OutputNetworkParameters, Interactive, Epsilon, Validate, ValidatePercentage, OutValues, OutDiff, PrintBehaviour, Threads
};

const std::map<std::string, CLIParameters> CLIParameterMap {
  {"--help",              CLIParameters::Help},
  {"-h",                  CLIParameters::Help},
  {"--input",             CLIParameters::InputFilePath},
  {"-i",                  CLIParameters::InputFilePath},
  {"--numberIn",          CLIParameters::NumberOfInputVariables},
  {"-ni",                 CLIParameters::NumberOfInputVariables},
  {"--numberOut",         CLIParameters::NumberOfOutputVariables},
  {"-no",                 CLIParameters::NumberOfOutputVariables},
  {"--epochs",            CLIParameters::NumberOfEpochs},
  {"-e",                  CLIParameters::NumberOfEpochs},
  {"--showProgress",      CLIParameters::ShowProgressDuringTraining},
  {"--inWeights",         CLIParameters::InputNetworkParameters},
  {"--outWeights",        CLIParameters::OutputNetworkParameters},
  {"--interactive",       CLIParameters::Interactive},
  {"--epsilon",           CLIParameters::Epsilon},
  {"--validate",          CLIParameters::Validate},
  {"--validatePercentage",CLIParameters::ValidatePercentage},
  {"--outValues",         CLIParameters::OutValues},
  {"--outDiff",           CLIParameters::OutDiff},
  {"--printBehaviour",    CLIParameters::PrintBehaviour},
  {"--threads",           CLIParameters::Threads},
  {"-t",                  CLIParameters::Threads}
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
  bool     ValidateAfterTraining {      DefaultValues::VALIDATE_AFTER_TRAINING };
  double   ValidationPercentage {       DefaultValues::VALIDATION_PERCENTAGE };
  FilePath OutputValuesFilePath {       DefaultValues::OUTPUT_VALUE };
  FilePath OutputDiffFilePath {         DefaultValues::OUTPUT_DIFF };
  bool     PrintBehaviour {             DefaultValues::PRINT_BEHAVIOUR };
  int32_t  NumberOfThreads {            DefaultValues::NUMBER_OF_THREADS };
};

}
