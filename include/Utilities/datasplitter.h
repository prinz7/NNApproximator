#pragma once

#include "Utilities/constants.h"

namespace Utilities {

class DataSplitter
{
public:
  [[nodiscard]]
  static std::pair<DataVector, DataVector> splitDataRandomly(DataVector const& inputData, double trainingPercentage);
  [[nodiscard]]
  static std::pair<DataVector, DataVector> splitDataWithThreshold(DataVector const& data, uint32_t thresholdVariable, TensorDataType threshold);
  [[nodiscard]]
  static BatchMap splitDataIntoBatches(DataVector const& data, uint32_t batchVariable);
};

}
