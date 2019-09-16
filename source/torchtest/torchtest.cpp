#include "MachineLearningTest/torchtest.h"

namespace TorchTest {

namespace {

const size_t NUMBER_OF_EPOCHS{1000};
const size_t SIZE_OF_TEST_DATA{20};
const double LOWER_BOUND_OF_TEST_RANGE{0};
const double UPPER_BOUND_OF_TEST_RANGE{5};
const double LEARNING_RATE{0.0001};

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

      if (x.item<TensorDataType>() == LOWER_BOUND_OF_TEST_RANGE && (epoch == 1 || epoch == NUMBER_OF_EPOCHS)) {
        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << "x: " << x.item<TensorDataType>() << std::endl;
        std::cout << "y: " << y.item<TensorDataType>() << std::endl;
        std::cout << "prediction: " << prediction.item<TensorDataType>() << std::endl;
        std::cout << "loss: " << loss.item<double>() << std::endl << std::endl;
      }
    }
  }

  bool stopTest = false;
  while (!stopTest) {
    std::cout << "Input x: ";

    std::string input;
    std::cin >> input;

    try {
      auto in = static_cast<TensorDataType>(std::stod(input));
      auto out = network.forward(torch::full(1, in));
      std::cout << "y: " << TargetFunction(in) << " prediction: " << out.item<TensorDataType>() << std::endl;
    } catch (std::exception&) {
      stopTest = true;
    }
  }
}

void TorchTest::initialize()
{
  TensorDataType stepSize = (UPPER_BOUND_OF_TEST_RANGE - LOWER_BOUND_OF_TEST_RANGE) / (SIZE_OF_TEST_DATA - 1);
  TensorDataType x = LOWER_BOUND_OF_TEST_RANGE;

  std::cout << "x values of the testdata: ";
  for (size_t i = 0; i < SIZE_OF_TEST_DATA; ++i) {
    std::cout << x << " ";
    auto xt = torch::full(1, x);
    auto yt = torch::full(1, TargetFunction(x));

    testData.emplace_back(xt, yt);
    x += stepSize;
  }
  std::cout << std::endl;

  std::cout << "Initialization finished!" << std::endl;
}

}
