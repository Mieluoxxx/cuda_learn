//
// Created by moguw on 24-12-8.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include "../utils.h"
#include "../../project/src/op/interface.h"


TEST(test_op, com_matrix_mul) {
  const int DSIZE = 8192;

  const float A_val = 3.0f;
  const float B_val = 2.0f;

  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;

  clock_t t0, t1, t2;

  double t1sum = 0.;
  double t2sum = 0.;

  // 开始计时
  t0 = clock();

  h_A = new float[DSIZE * DSIZE];
  h_B = new float[DSIZE * DSIZE];
  h_C = new float[DSIZE * DSIZE];

  for (int i = 0; i < DSIZE * DSIZE; i++) {
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // 计时结束
  t1 = clock();
  t1sum = (double) (t1 - t0) / CLOCKS_PER_SEC;
  printf("Init took %f seconds. Begin compute\n", t1sum);


  // 为设备端分配内存，并将主机端数据复制到设备端
  // 分配 GPU 内存，并检查错误
  checkCudaError(cudaMalloc((void **) &d_A, DSIZE * DSIZE * sizeof(float)), "cudaMalloc for d_a");
  checkCudaError(cudaMalloc((void **) &d_B, DSIZE * DSIZE * sizeof(float)), "cudaMalloc for d_b");
  checkCudaError(cudaMalloc((void **) &d_C, DSIZE * DSIZE * sizeof(float)), "cudaMalloc for d_c");

  checkCudaError(cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy for d_a");
  checkCudaError(cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy for d_b");

  mop::get_com_matrix_mul_op(mbase::DeviceType::Device)(d_A, d_B, d_C, DSIZE, DSIZE);


  // CUDA处理的第二步完成（计算完成）
  // 将结果从设备端复制回主机端
  checkCudaError(cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy for d_b");

  // 计算完成，记录时间
  t2 = clock();
  t2sum = ((double) (t2 - t1)) / CLOCKS_PER_SEC; // 计算计算时间
  printf("Done. Compute took %f seconds\n", t2sum);


  // CUDA处理的第三步完成（结果返回主机）
  for (int i = 0; i < DSIZE * DSIZE; i++) // 遍历所有元素进行验证
    if (h_C[i] != A_val * B_val * DSIZE) {
      // 如果计算结果不正确，输出错误信息
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
    }
  printf("Success!\n");

  // 释放内存
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
