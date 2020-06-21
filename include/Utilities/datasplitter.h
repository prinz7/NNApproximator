#pragma once

#include "Utilities/constants.h"

namespace Utilities {

class DataSplitter
{
public:
  /*
   * Splits the data randomly into two vectors with the given probability.
   */
  [[nodiscard]]
  static std::pair<DataVector, DataVector> splitDataRandomly(DataVector const& inputData, double trainingPercentage);
  /*
   * Splits the data into two vectors with the given threshold.
   */
  [[nodiscard]]
  static std::pair<DataVector, DataVector> splitDataWithThreshold(DataVector const& data, uint32_t thresholdVariable, TensorDataType threshold);
  /*
   * Splits the data into batches with the given batch variable.
   */
  [[nodiscard]]
  static BatchMap splitDataIntoBatches(DataVector const& data, uint32_t batchVariable);
};

}
