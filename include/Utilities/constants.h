#pragma once

#include <torch/torch.h>
#include <vector>

using TensorDataType = float;
using DataVector = std::vector<std::pair<torch::Tensor, torch::Tensor>>;
using MinMaxVector = std::vector<std::pair<TensorDataType, TensorDataType>>;

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
