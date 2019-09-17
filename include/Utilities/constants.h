#pragma once

#include <torch/torch.h>
#include <vector>

using TensorDataType = double;
using DataVector = std::vector<std::pair<torch::Tensor, torch::Tensor>>;
