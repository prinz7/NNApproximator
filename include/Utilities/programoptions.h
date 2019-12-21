#pragma once

#include <torch/torch.h>

#include <map>
#include <string>

#include "Utilities/constants.h"

namespace Utilities {

namespace DefaultValues {

const FilePath                INPUT_DATA_FILE_PATH = {};
const FilePath                INPUT_NETWORK_PARAMETERS = {};
const FilePath                OUTPUT_NETWORK_PARAMETERS = {};
const uint32_t                NUMBER_OF_INPUT_VARIABLES = 1;
const uint32_t                NUMBER_OF_OUTPUT_VARIABLES = 1;
const uint32_t                NUMBER_OF_EPOCHS = 10;
const bool                    SHOW_PROGRESS_DURING_TRAINING = true;
const bool                    INTERACTIVE_MODE = false;
const double                  EPSILON = 1.0;
const bool                    LOG_SCALING = false;
const bool                    SQRT_SCALING = false;
const bool                    LOG_LIN_SCALING = false;
const bool                    LOG_SQRT_SCALING = false;
const uint32_t                MIXED_SCALING_INPUT_VARIABLE = 0;
const TensorDataType          MIXED_SCALING_THRESHOLD = 0.0;
const bool                    VALIDATE_AFTER_TRAINING = false;
const double                  VALIDATION_PERCENTAGE = 30.0;
const FilePath                OUTPUT_VALUE = {};
const FilePath                OUTPUT_DIFF = {};
const bool                    PRINT_BEHAVIOUR = false;
const int32_t                 NUMBER_OF_THREADS = torch::get_num_threads();
const FilePath                INPUT_MIN_MAX_FILE_PATH = {};
const FilePath                OUTPUT_MIN_MAX_FILE_PATH = {};
const double                  LEARN_RATE = 0.001;
const TimeoutDuration         MAX_EXECUTION_TIME = std::chrono::duration_cast<TimeoutDuration>(std::chrono::hours(24 * 7)); // TODO change to std::chrono::weeks when switching to C++20
const uint32_t                NUMBER_OF_DETERIORATIONS = 0;
const FilePath                PROGRESS_FILE_PATH = {};
const std::optional<uint64_t> RANDOM_GENERATOR_SEED = std::nullopt;

const std::string CLI_HELP_TEXT = {
  std::string("List of possible commandline parameters:\n") +
  "--help | -h                        : Output this text message.\n" +
  "--input <filepath> | -i <filepath> : Use content of <filepath> for the input data.\n" +
  "--numberIn X | -ni X               : Sets the number of input variables to X. Default: " + std::to_string(NUMBER_OF_INPUT_VARIABLES) + "\n" +
  "--numberOut X | -no X              : Sets the number of output variables to X. Default: " + std::to_string(NUMBER_OF_OUTPUT_VARIABLES) + "\n" +
  "--epochs X | -e X                  : Sets the minimum number of epochs (how many times the data is used for training). Default: " + std::to_string(NUMBER_OF_EPOCHS) + "\n" +
  "--showProgress <bool>              : Activate or deactivate display of progress and eta of the training. Default: " + (SHOW_PROGRESS_DURING_TRAINING ? "true" : "false") + "\n" +
  "--inWeights <filepath>             : If set loads the weights in the file for the network in the initialization phase.\n" +
  "--outWeights <filepath>            : If set saves the weights of the network to the specified file after the training phase.\n" +
  "--interactive                      : If set activated the interactive mode after the training to test user input on the neural network.\n" +
  "--epsilon <double>                 : If set continues training after the last epoch until the improvement of the mean squarred error is less than the set epsilon. Default: " + std::to_string(EPSILON) + "\n" +
  "--logScaling                       : If set scales the output values logarithmic for the neural network. Does not work together with other scaling options.\n" +
  "--sqrtScaling                      : If set scales the output values with the square root function for the neural network. Does not work together with other scaling options.\n" +
  "--logLinScaling X <double>         : If set scales the output logarithmic if input X [1, ..] is below or equal the given value and no scaling above it. Does not work together with other scaling options.\n" +
  "--logSqrtScaling X <double>        : If set scales the output logarithmic if input X [1, ..] is below or equal the given value and sqrt scaling above it. Does not work together with other scaling options.\n" +
  "--validate                         : If set splits the data set in a training and validation set. After the training the network is tested with the validation set.\n" +
  "--validatePercentage <double>      : Sets the percentage of the data, which is only used for validation and not for training. Value should be between 0 and 100. Default: " + std::to_string(VALIDATION_PERCENTAGE) + "\n" +
  "--outValues <filepath>             : If set saves the output of the neural network for all input values to the specified file.\n" +
  "--outDiff <filepath>               : If set saves the difference of the output of the neural network and given input values to the specified file.\n" +
  "--printBehaviour                   : If set outputs the behaviour of the neural network to the console for the given input values.\n" +
  "--threads X | -t X                 : Sets the number of used threads to X. Default value depends on the given system. Default value of the current system: " + std::to_string(NUMBER_OF_THREADS) + "\n" +
  "--inMinMax <filepath>              : If set uses the data in the given file to use as min/max values for normalization.\n" +
  "--outMinMax <filepath>             : If set saves the used min/max values to the given file.\n" +
  "--learnRate <double>               : Sets the learning rate of the statistical gradient descent. Default: " + std::to_string(LEARN_RATE) + "\n" +
  "--timeoutInMinutes X               : Sets the timeout of the program to X minutes. Default: 1 week.\n" +
  "--timeoutInHours X                 : Sets the timeout of the program to X hours. Default: 1 week.\n" +
  "--numberOfDeteriorations X         : Sets the number of epochs in a row in which the improvement can be worse than the set epsilon without stopping. Default: " + std::to_string(NUMBER_OF_DETERIORATIONS) + "\n" +
  "--saveProgress <filepath>          : If set saves the progress in a CSV file at the specified path.\n" +
  "--seed <uint64>                    : Sets the seed of the random number generator, which is used for initializing the network parameters.\n"
};

}

enum class CLIParameters
{
  Help, InputFilePath, NumberOfInputVariables, NumberOfOutputVariables, NumberOfEpochs, ShowProgressDuringTraining, InputNetworkParameters,
  OutputNetworkParameters, Interactive, Epsilon, LogScaling, SqrtScaling, LogLinScaling, LogSqrtScaling, Validate, ValidatePercentage, OutValues,
  OutDiff, PrintBehaviour, Threads, InputMinMax, OutputMinMax, LearnRate, TimeoutMinutes, TimeoutHours, NumberOfDeteriorations, SaveProgress,
  Seed
};

const std::map<std::string, CLIParameters> CLIParameterMap {
  {"--help",                  CLIParameters::Help},
  {"-h",                      CLIParameters::Help},
  {"--input",                 CLIParameters::InputFilePath},
  {"-i",                      CLIParameters::InputFilePath},
  {"--numberIn",              CLIParameters::NumberOfInputVariables},
  {"-ni",                     CLIParameters::NumberOfInputVariables},
  {"--numberOut",             CLIParameters::NumberOfOutputVariables},
  {"-no",                     CLIParameters::NumberOfOutputVariables},
  {"--epochs",                CLIParameters::NumberOfEpochs},
  {"-e",                      CLIParameters::NumberOfEpochs},
  {"--showProgress",          CLIParameters::ShowProgressDuringTraining},
  {"--inWeights",             CLIParameters::InputNetworkParameters},
  {"--outWeights",            CLIParameters::OutputNetworkParameters},
  {"--interactive",           CLIParameters::Interactive},
  {"--epsilon",               CLIParameters::Epsilon},
  {"--logScaling",            CLIParameters::LogScaling},
  {"--sqrtScaling",           CLIParameters::SqrtScaling},
  {"--logLinScaling",         CLIParameters::LogLinScaling},
  {"--logSqrtScaling",        CLIParameters::LogSqrtScaling},
  {"--validate",              CLIParameters::Validate},
  {"--validatePercentage",    CLIParameters::ValidatePercentage},
  {"--outValues",             CLIParameters::OutValues},
  {"--outDiff",               CLIParameters::OutDiff},
  {"--printBehaviour",        CLIParameters::PrintBehaviour},
  {"--threads",               CLIParameters::Threads},
  {"-t",                      CLIParameters::Threads},
  {"--inMinMax",              CLIParameters::InputMinMax},
  {"--outMinMax",             CLIParameters::OutputMinMax},
  {"--learnRate",             CLIParameters::LearnRate},
  {"--timeoutInMinutes",      CLIParameters::TimeoutMinutes},
  {"--timeoutInHours",        CLIParameters::TimeoutHours},
  {"--numberOfDeteriorations",CLIParameters::NumberOfDeteriorations},
  {"--saveProgress",          CLIParameters::SaveProgress},
  {"--seed",                  CLIParameters::Seed}
};

class ProgramOptions
{
public:
  FilePath                InputDataFilePath {          DefaultValues::INPUT_DATA_FILE_PATH };
  FilePath                InputNetworkParameters {     DefaultValues::INPUT_NETWORK_PARAMETERS };
  FilePath                OutputNetworkParameters {    DefaultValues::OUTPUT_NETWORK_PARAMETERS };
  uint32_t                NumberOfInputVariables {     DefaultValues::NUMBER_OF_INPUT_VARIABLES };
  uint32_t                NumberOfOutputVariables {    DefaultValues::NUMBER_OF_OUTPUT_VARIABLES };
  uint32_t                NumberOfEpochs {             DefaultValues::NUMBER_OF_EPOCHS };
  bool                    ShowProgressDuringTraining { DefaultValues::SHOW_PROGRESS_DURING_TRAINING };
  bool                    InteractiveMode {            DefaultValues::INTERACTIVE_MODE };
  double                  Epsilon {                    DefaultValues::EPSILON };
  bool                    LogScaling {                 DefaultValues::LOG_SCALING };
  bool                    SqrtScaling {                DefaultValues::SQRT_SCALING };
  bool                    LogLinScaling {              DefaultValues::LOG_LIN_SCALING };
  bool                    LogSqrtScaling {             DefaultValues::LOG_SQRT_SCALING };
  uint32_t                MixedScalingInputVariable {  DefaultValues::MIXED_SCALING_INPUT_VARIABLE };
  TensorDataType          MixedScalingThreshold {      DefaultValues::MIXED_SCALING_THRESHOLD };
  bool                    ValidateAfterTraining {      DefaultValues::VALIDATE_AFTER_TRAINING };
  double                  ValidationPercentage {       DefaultValues::VALIDATION_PERCENTAGE };
  FilePath                OutputValuesFilePath {       DefaultValues::OUTPUT_VALUE };
  FilePath                OutputDiffFilePath {         DefaultValues::OUTPUT_DIFF };
  bool                    PrintBehaviour {             DefaultValues::PRINT_BEHAVIOUR };
  int32_t                 NumberOfThreads {            DefaultValues::NUMBER_OF_THREADS };
  FilePath                InputMinMaxFilePath {        DefaultValues::INPUT_MIN_MAX_FILE_PATH };
  FilePath                OutputMinMaxFilePath {       DefaultValues::OUTPUT_MIN_MAX_FILE_PATH };
  double                  LearnRate {                  DefaultValues::LEARN_RATE };
  TimeoutDuration         MaxExecutionTime {           DefaultValues::MAX_EXECUTION_TIME };
  uint32_t                NumberOfDeteriorations {     DefaultValues::NUMBER_OF_DETERIORATIONS };
  FilePath                SaveProgressFilePath {       DefaultValues::PROGRESS_FILE_PATH };
  std::optional<uint64_t> RNGSeed {                    DefaultValues::RANDOM_GENERATOR_SEED };
};

}
