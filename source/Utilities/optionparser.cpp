#include "Utilities/optionparser.h"

#include <iostream>

namespace Utilities {

[[nodiscard]]
static inline bool ConvertStringToBool(std::string const& str)
{
  return !(str == "0" || str == "false" || str == "False" || str == "FALSE");
}

std::optional<ProgramOptions> OptionParser::ParseCommandLineParameters(int argc, char* argv[])
{
  auto options = Utilities::ProgramOptions();
  bool validationPercentageSet = false;

  for (int i = 1; i < argc; ++i) {
    std::string inputString (argv[i]);

    auto inputParameter = CLIParameterMap.find(inputString);
    if (inputParameter == CLIParameterMap.end()) {
      std::cout << "Unknown commandline parameter: " << inputString << "\n"
                << "Use --help to see available commands." << std::endl;
      return std::nullopt;
    }

    switch (inputParameter->second) {
      case CLIParameters::Help:
        std::cout << DefaultValues::CLI_HELP_TEXT << std::endl;
        exit(0);
      case CLIParameters::InputFilePath:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.InputDataFilePath = std::string(argv[++i]);
        break;
      case CLIParameters::NumberOfInputVariables:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.NumberOfInputVariables = std::stoul(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::NumberOfOutputVariables:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.NumberOfOutputVariables = std::stoul(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::NumberOfEpochs:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.NumberOfEpochs = std::stoul(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::ShowProgressDuringTraining:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.ShowProgressDuringTraining = ConvertStringToBool(argv[++i]);
        break;
      case CLIParameters::InputNetworkParameters:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.InputNetworkParameters = std::string(argv[++i]);
        break;
      case CLIParameters::OutputNetworkParameters:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.OutputNetworkParameters = std::string(argv[++i]);
        break;
      case CLIParameters::Interactive:
        options.InteractiveMode = true;
        break;
      case CLIParameters::Epsilon:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.Epsilon = std::stod(std::string(argv[++i]));
        } catch (std::exception const&) {
          std::cout << "Could not parse " << std::string(argv[i]) << " to double." << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::LogScaling:
        if (options.SqrtScaling || options.LogLinScaling || options.LogSqrtScaling) {
          std::cout << "Only one scaling option is allowed." << std::endl;
          return std::nullopt;
        }
        options.LogScaling = true;
        break;
      case CLIParameters::SqrtScaling:
        if (options.LogScaling || options.LogLinScaling || options.LogSqrtScaling) {
          std::cout << "Only one scaling option is allowed." << std::endl;
          return std::nullopt;
        }
        options.SqrtScaling = true;
        break;
      case CLIParameters::LogLinScaling:
        if (i + 2 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        if (options.LogScaling || options.SqrtScaling || options.LogSqrtScaling) {
          std::cout << "Only one scaling option is allowed." << std::endl;
          return std::nullopt;
        }
        try {
          options.MixedScalingInputVariable = std::stoi(std::string(argv[++i]));
        } catch (std::exception const&) {
          std::cout << "Could not parse " << std::string(argv[i]) << " to int." << std::endl;
          return std::nullopt;
        }
        try {
          options.MixedScalingThreshold = std::stod(std::string(argv[++i]));
        } catch (std::exception const&) {
          std::cout << "Could not parse " << std::string(argv[i]) << " to double." << std::endl;
          return std::nullopt;
        }
        options.LogLinScaling = true;
        break;
      case CLIParameters::LogSqrtScaling:
        if (i + 2 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        if (options.LogScaling || options.SqrtScaling || options.LogLinScaling) {
          std::cout << "Only one scaling option is allowed." << std::endl;
          return std::nullopt;
        }
        try {
          options.MixedScalingInputVariable = std::stoi(std::string(argv[++i]));
        } catch (std::exception const&) {
          std::cout << "Could not parse " << std::string(argv[i]) << " to int." << std::endl;
          return std::nullopt;
        }
        try {
          options.MixedScalingThreshold = std::stod(std::string(argv[++i]));
        } catch (std::exception const&) {
          std::cout << "Could not parse " << std::string(argv[i]) << " to double." << std::endl;
          return std::nullopt;
        }
        options.LogSqrtScaling = true;
        break;
      case CLIParameters::Validate:
        options.ValidateAfterTraining = true;
        break;
      case CLIParameters::ValidatePercentage:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.ValidationPercentage = std::stod(std::string(argv[++i]));
          validationPercentageSet = true;
        } catch (std::exception const&) {
          std::cout << "Could not parse " << std::string(argv[i]) << " to double." << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::OutValues:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.OutputValuesFilePath = std::string(argv[++i]);
        break;
      case CLIParameters::OutDiff:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.OutputDiffFilePath = std::string(argv[++i]);
        break;
      case CLIParameters::OutRelativeDiff:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.OutputRelativeDiffFilePath = std::string(argv[++i]);
        break;
      case CLIParameters::PrintBehaviour:
        options.PrintBehaviour = true;
        break;
      case CLIParameters::Threads:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.NumberOfThreads = std::stoi(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::InputMinMax:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.InputMinMaxFilePath = std::string(argv[++i]);
        break;
      case CLIParameters::OutputMinMax:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.OutputMinMaxFilePath = std::string(argv[++i]);
        break;
      case CLIParameters::LearnRate:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.LearnRate = std::stod(std::string(argv[++i]));
        } catch (std::exception const&) {
          std::cout << "Could not parse " << std::string(argv[i]) << " to double." << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::TimeoutMinutes:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          auto minutes = std::stoul(argv[++i]);
          options.MaxExecutionTime = std::chrono::duration_cast<TimeoutDuration>(std::chrono::minutes(minutes));
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::TimeoutHours:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          auto hours = std::stoul(argv[++i]);
          options.MaxExecutionTime = std::chrono::duration_cast<TimeoutDuration>(std::chrono::hours(hours));
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::NumberOfDeteriorations:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.NumberOfDeteriorations = std::stoul(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::SaveProgress:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        options.SaveProgressFilePath = std::string(argv[++i]);
        break;
      case CLIParameters::Seed:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.RNGSeed = std::make_optional(std::stoul(argv[++i]));
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::NumberOfLayers:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.NumberOfLayers = std::stoul(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::NumberOfNodes:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.NumberOfNodesPerLayer = std::stoul(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::BatchVariable:
        if (i + 1 >= argc) {
          std::cout << "Not enough parameters after " << inputString << std::endl;
          return std::nullopt;
        }
        try {
          options.BatchVariable = std::stoul(argv[++i]);
        } catch (const std::invalid_argument& e) {
          std::cout << "Could not convert " << std::string(argv[i]) << " to integer. Reason: " << e.what() << std::endl;
          return std::nullopt;
        } catch (const std::out_of_range& e) {
          std::cout << std::string(argv[i]) << " is out of range. Error: " << e.what() << std::endl;
          return std::nullopt;
        }
        break;
      case CLIParameters::DebugOutput:
        options.DebugOutput = true;
        break;
    }
  }

  // Sanity checks:
  if (options.MixedScalingInputVariable > options.NumberOfInputVariables) {
    std::cout << "Input variable " << options.MixedScalingInputVariable << " is not usable for mixed scaling, because there are only " << options.NumberOfInputVariables << " variables available." << std::endl;
    return std::nullopt;
  }

  if (options.NumberOfLayers == 0) {
    std::cout << "Number of layers should be > 0." << std::endl;
    return std::nullopt;
  }

  if (options.NumberOfNodesPerLayer == 0) {
    std::cout << "Number of nodes per layer should be > 0." << std::endl;
    return std::nullopt;
  }

  if (options.NumberOfLayers * options.NumberOfNodesPerLayer > MaxNumberOfNodes) {
    std::cout << "Total number of nodes should not exceed " << MaxNumberOfNodes << std::endl;
    return std::nullopt;
  }

  if (options.BatchVariable.has_value() && options.BatchVariable.value() > options.NumberOfInputVariables) {
    std::cout << "Invalid batch variable: " << options.BatchVariable.value() << " -- number of input variables: " << options.NumberOfInputVariables << std::endl;
    return std::nullopt;
  }

  if (options.NumberOfThreads < 1) {
    std::cout << "Invalid number of threads: " << options.NumberOfThreads << ". Please input a number > 0." << std::endl;
    return std::nullopt;
  }

  if (options.LearnRate <= 0.0) {
    std::cout << "Invalid learning rate: " << options.LearnRate << ". Please input a number > 0." << std::endl;
    return std::nullopt;
  }

  // Warnings:
  if (validationPercentageSet && !options.ValidateAfterTraining) {
    std::cout << "[Warning] A validation percentage was set, but the validation mode is not active! Activate validation with --validate" << std::endl;
  }

  if (!options.InteractiveMode && !options.PrintBehaviour && options.OutputDiffFilePath == DefaultValues::OUTPUT_DIFF &&
      options.OutputRelativeDiffFilePath == DefaultValues::OUTPUT_RELATIVE_DIFF && options.OutputMinMaxFilePath == DefaultValues::OUTPUT_MIN_MAX_FILE_PATH &&
      options.OutputNetworkParameters == DefaultValues::OUTPUT_NETWORK_PARAMETERS && options.OutputValuesFilePath == DefaultValues::OUTPUT_VALUE) {
    std::cout << "[Warning] No option was set to output something. For available commands try --help" << std::endl;
  }

  if (options.MaxExecutionTime > std::chrono::hours(24 * 7)) {
    std::cout << "[Warning] The timeout is set to a very long time (> 1 week). If the execution is interrupted, all progress is lost." << std::endl;
  }

  if (options.Epsilon < 0.0) {
    std::cout << "[Warning] With a negative epsilon, the probability is high that the program runs until the set timeout (even if no progress is made)." << std::endl;
  }

  // Change index range from [1, ...] to [0, ...]
  options.MixedScalingInputVariable--;
  (*options.BatchVariable)--;

  return std::make_optional(options);
}

}
