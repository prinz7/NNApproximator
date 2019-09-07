#include "MachineLearningTest/torchtest.h"

namespace TorchTest {

namespace {

const size_t NUMBER_OF_EPOCHS{10000};
const size_t SIZE_OF_TEST_DATA{2};
const double LOWER_BOUND_OF_TEST_RANGE{0};
const double UPPER_BOUND_OF_TEST_RANGE{4};
const double LEARNING_RATE{0.001};

}

TensorDataType TorchTest::TargetFunction(TensorDataType x)
{
  return 10 + 7 * x + 5 * x * x;
}

TorchTest::TorchTest()
{
  initialize();
}

void TorchTest::run()
{
  torch::optim::SGD optimizer(network.parameters(), LEARNING_RATE);

  for (size_t epoch = 1; epoch <= NUMBER_OF_EPOCHS; ++epoch) {
    for (auto [x, y] : testData) {
      optimizer.zero_grad();

      auto prediction = network.forward(x);



//      prediction = prediction.toType(torch::ScalarType::Long);
//      auto target = y.toType(torch::ScalarType::Long);
      auto loss = torch::mse_loss(prediction, y);
//      auto loss = torch::kl_div(prediction, y);
//      auto loss = torch::nll_loss(prediction, y);
      loss.backward();
      optimizer.step();

      if (epoch == 1 || epoch == NUMBER_OF_EPOCHS) {
        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << "x: " << x.item<TensorDataType>() << std::endl;
        std::cout << "y: " << y.item<TensorDataType>() << std::endl;
        std::cout << "prediction: " << prediction.item<TensorDataType>() << std::endl;
        std::cout << "loss: " << loss.item<double>() << std::endl;
      }
    }
  }

  // TODO interactive testing of the network
}

void TorchTest::initialize()
{
  TensorDataType stepSize = (UPPER_BOUND_OF_TEST_RANGE - LOWER_BOUND_OF_TEST_RANGE) / (SIZE_OF_TEST_DATA - 1);
  TensorDataType x = LOWER_BOUND_OF_TEST_RANGE;

  for (size_t i = 0; i < SIZE_OF_TEST_DATA; ++i) {
    auto xt = torch::full((1), x, torch::ScalarType::Float);
    auto yt = torch::full((1), TargetFunction(x), torch::ScalarType::Float);

    testData.emplace_back(xt, yt);
    x += stepSize;
  }

  std::cout << "Initialization finished!" << std::endl;
}

}
