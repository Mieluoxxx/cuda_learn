include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)

# 显式指定源文件
set(DIR_TEST
        ${CMAKE_SOURCE_DIR}/test/main.cpp
        ${CMAKE_SOURCE_DIR}/test/utils.cpp
)

set(DIR_TEST_OP
        ${CMAKE_SOURCE_DIR}/test/op_test/op_test.cpp
        ${CMAKE_SOURCE_DIR}/test/op_test/op_com_matmul.cpp
        ${CMAKE_SOURCE_DIR}/test/op_test/op_stencil_1d.cpp
)

add_executable(test_main ${DIR_TEST} ${DIR_TEST_OP})

target_include_directories(test_main
        PUBLIC
        ${glog_INCLUDE_DIR}
        ${GTest_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/test
)

target_link_directories(test_main
        PUBLIC
        ${PROJECT_SOURCE_DIR}/lib
)

target_link_libraries(test_main cudalearn)

set_target_properties(test_main PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_target_properties(test_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
