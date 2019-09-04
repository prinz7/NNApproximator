cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(SET CMP0054 NEW)

project(TorchTest)

find_package(Torch REQUIRED)

add_executable(TorchTest source/main.cpp)

target_link_libraries(TorchTest "${TORCH_LIBRARIES}")

set_property(TARGET TorchTest PROPERTY CXX_STANDARD 11)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
#if (MSVC)
#    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#    add_custom_command(TARGET TorchTest
#            POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            ${TORCH_DLLS}
#            $<TARGET_FILE_DIR:TorchTest>)
#endif (MSVC)
