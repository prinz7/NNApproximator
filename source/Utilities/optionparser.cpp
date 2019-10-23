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
    }
  }

  return std::make_optional(options);
}

}
