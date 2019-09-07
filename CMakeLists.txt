cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(TorchTest)
add_executable(TorchTest "")
set_property(TARGET TorchTest PROPERTY CXX_STANDARD 17)

target_include_directories(TorchTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

list(APPEND CMAKE_PREFIX_PATH "libs/libtorch")
find_package(Torch REQUIRED)

target_link_libraries(TorchTest "${TORCH_LIBRARIES}")

add_subdirectory(source)
