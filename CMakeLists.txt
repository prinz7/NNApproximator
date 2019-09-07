cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

list(APPEND CMAKE_PREFIX_PATH "libs/libtorch")

project(TorchTest)

find_package(Torch REQUIRED)

add_executable(TorchTest source/main.cpp)

target_link_libraries(TorchTest "${TORCH_LIBRARIES}")

set_property(TARGET TorchTest PROPERTY CXX_STANDARD 17)
