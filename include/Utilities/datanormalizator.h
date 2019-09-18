#pragma once

#include "Utilities/constants.h"

namespace Utilities {

class DataNormalizator
{
public:
  static void Normalize(DataVector& data, std::pair<MinMaxVector, MinMaxVector>& minMaxVectors, TensorDataType newMinValue = -0.5, TensorDataType newMaxValue = 0.5);
  static void Denormalize(torch::Tensor& tensor, MinMaxVector const& minMaxVector, TensorDataType oldMinValue = -0.5, TensorDataType oldMaxValue = 0.5);
};

}
