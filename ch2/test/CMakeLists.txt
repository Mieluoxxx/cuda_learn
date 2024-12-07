include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)
aux_source_directory(../test/op_test DIR_TEST_OP)

add_executable(test_main ${DIR_TEST} ${DIR_TEST_OP})

target_include_directories(test_main
        PUBLIC
        ${glog_INCLUDE_DIR}
        ${GTest_INCLUDE_DIR}
        ../project/include
)

target_link_directories(test_main
        PUBLIC
        ${PROJECT_SOURCE_DIR}/lib
)

target_link_libraries(test_main cudalearn)

set_target_properties(test_main PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_target_properties(test_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
