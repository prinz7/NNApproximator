#pragma once

#include <torch/torch.h>
#include <vector>

// Forward declarations:

class LearnProgressDataSet;

// Type definitions & constants:

using TensorDataType = double;
const torch::ScalarType TORCH_DATA_TYPE = torch::kDouble;

using DataVector = std::vector<std::pair<torch::Tensor, torch::Tensor>>;
using MinMaxVector = std::vector<std::pair<TensorDataType, TensorDataType>>;

using ProgressVector = std::vector<LearnProgressDataSet>;
const std::string LEARN_PROGRESS_FILE_HEADER = "Epoch, R2Score, MeanSquaredError, ElapsedTimeInMS";

// Data classes:

class LearnProgressDataSet
{
public:
  uint32_t epoch;
  double r2Score;
  double meanSquaredError;
  uint64_t elapsedTimeInMS;
};

// Helper functions:

template<class DurationIn, class FirstDuration, class...RestDurations>
static std::string formatDuration(DurationIn d)
{
  auto val = std::chrono::duration_cast<FirstDuration>(d);

  std::string out = std::to_string(val.count());
  if (out.size() == 1) {
    out = "0" + out;
  }

  if constexpr(sizeof...(RestDurations) > 0) {
    out += ":" + formatDuration<DurationIn, RestDurations...>(d - val);
  }

  return out;
}

template<class DurationIn>
static std::string formatDuration(DurationIn) { return {}; } // recursion termination
