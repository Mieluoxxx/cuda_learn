cmake_minimum_required(VERSION 3.20)

# 设置CUDA编译器的路径
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")

# 定义项目名称和使用的编程语言
project(ch4 CXX CUDA)

# 包含自定义的CUDA CMake模块
include(cmake/cuda.cmake)

# 设置C++标准为17，并确保是必需的
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库文件的输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 查找并设置Google Test库
find_package(GTest REQUIRED)
# 查找并设置glog库
find_package(glog REQUIRED)

# 自动查找目录下的所有源文件，并存储在变量中
aux_source_directory(project/src/ DIR_MAIN)
aux_source_directory(project/src/op/gpu/ DIR_KERNEL_CUDA)
aux_source_directory(project/src/op/ DIR_OP)
aux_source_directory(project/src/base/ DIR_BASE)

# 创建一个名为cudalearn的共享库，包含所有指定的源文件
add_library(cudalearn SHARED ${DIR_MAIN} ${DIR_KERNEL_CUDA} ${DIR_OP} ${DIR_BASE})

# 为目标cudalearn链接所需的库
target_link_libraries(
        cudalearn
        PUBLIC
        glog::glog          # glog库
        GTest::gtest        # Google Test的gtest库
        GTest::gtest_main   # Google Test的gtest_main库
        pthread             # POSIX线程库
        cudart              # CUDA运行时库
)

# 为cudalearn指定头文件目录
target_include_directories(cudalearn PUBLIC
        ${glog_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/project/include
        ${PROJECT_SOURCE_DIR}/project/include/base
        ${PROJECT_SOURCE_DIR}/project/src/op
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 为cudalearn设置CUDA分离编译属性
set_target_properties(cudalearn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(test)

# ------------- 编译主程序 ---------------

## 定义一个变量link_exb_lib，包含需要链接的外部库
#set(link_exb_lib glog::glog GTest::gtest)
#
## 创建一个名为demo的可执行文件，其源文件为project/src/main.cpp
#add_executable(demo project/src/main.cpp)
#
## 为demo指定头文件目录
#target_include_directories(demo PUBLIC
#        ${glog_INCLUDE_DIR}
#        ${GTest_INCLUDE_DIR}
#        ${PROJECT_SOURCE_DIR}/project/include
#)
#
## 为demo指定库目录
#target_link_directories(demo PUBLIC ${PROJECT_SOURCE_DIR}/lib)
#
## 为demo链接外部库和cudalearn库
#target_link_libraries(demo
#        PUBLIC
#        ${link_exb_lib}
#        cudalearn
#)
#
## 设置demo的目标属性，包括工作目录和CUDA分离编译属性
#set_target_properties(demo PROPERTIES
#        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
#        CUDA_SEPARABLE_COMPILATION ON
#)