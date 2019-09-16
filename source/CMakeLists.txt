target_sources(TorchTest
    PRIVATE
        main.cpp
)

add_subdirectory(NeuralNetwork)
add_subdirectory(torchtest)
add_subdirectory(Utilities)
