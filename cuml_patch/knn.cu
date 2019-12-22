/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/cumlHandle.hpp"

#include <cuml/neighbors/knn.hpp>

#include "ml_mg_utils.h"

#include "label/classlabels.h"
#include "selection/knn.h"

#include <cuda_runtime.h>
#include "cuda_utils.h"

#include <sstream>
#include <vector>

// BEGIN SWEET KNN HEADERS
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include "cublas_v2.h"

using namespace std;
// END SWEET KNN HEADERS

namespace ML {

void brute_force_knn(cumlHandle &handle, std::vector<float *> &input,
                     std::vector<int> &sizes, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k, bool rowMajorIndex,
                     bool rowMajorQuery) {
  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors must be the same size");

  std::vector<cudaStream_t> int_streams = handle.getImpl().getInternalStreams();

  MLCommon::Selection::brute_force_knn(
    input, sizes, D, search_items, n, res_I, res_D, k,
    handle.getImpl().getDeviceAllocator(), handle.getImpl().getStream(),
    int_streams.data(), handle.getImpl().getNumInternalStreams(), rowMajorIndex,
    rowMajorQuery);
}

// BEGIN SWEET KNN CODE
typedef struct Point2Rep {
  int repIndex;
  float dist2rep;
} P2R;

typedef struct IndexAndDist {
  int index;
  float dist;
} IndexDist;

typedef struct repPoint_static {
  float maxdist = 0.0f;
  float mindist = FLT_MAX;
  uint npoints = 0;
  uint noreps = 0;
  float kuboundMax = 0.0f;
} R2all_static;

typedef struct repPoint_static_dev {
  float maxdist;
  float mindist;
  uint npoints;
  uint noreps;
  float kuboundMax;
} R2all_static_dev;

typedef struct repPoint_dynamic_p {
  int *memberID;
  IndexDist *sortedmembers;
  float *kubound;
  IndexDist *replist;
} R2all_dyn_p;

struct sort_dec {
  bool operator()(const IndexDist &left, const IndexDist &right) {
    return left.dist > right.dist;
  }
};

struct sort_inc {
  bool operator()(const IndexDist &left, const IndexDist &right) {
    return left.dist < right.dist;
  }
};

void timePoint(struct timespec &T1) { clock_gettime(CLOCK_REALTIME, &T1); }

float timeLen(struct timespec &T1, struct timespec &T2) {
  return T2.tv_sec - T1.tv_sec + (T2.tv_nsec - T1.tv_nsec) / 1.e9;
}

__device__ float Edistance_128(float *a, float *b, int dim) {
  float distance = 0.0f;
  float4 *A = (float4 *)a;
  float4 *B = (float4 *)b;
  float tmp = 0.0f;
  for (int i = 0; i < int(dim / 4); i++) {
    float4 a_local = A[i];
    float4 b_local = __ldg(&B[i]);
    tmp = a_local.x - b_local.x;
    distance += tmp * tmp;
    tmp = a_local.y - b_local.y;
    distance += tmp * tmp;
    tmp = a_local.z - b_local.z;
    distance += tmp * tmp;
    tmp = a_local.w - b_local.w;
    distance += tmp * tmp;
  }
  for (int i = int(dim / 4) * 4; i < dim; i++) {
    tmp = (a[i]) - (b[i]);
    distance += tmp * tmp;
  }
  return sqrt(distance);
}

__host__ __device__ float Edistance(float *A, float *B, int dim) {
  float distance = 0.0f;
  for (int i = 0; i < dim; i++) {
    float tmp = A[i] - B[i];
    distance += tmp * tmp;
  }
  return sqrt(distance);
}

__device__ __forceinline__ int F2I(float floatVal) {
  int intVal = __float_as_int(floatVal);
  return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

__device__ __forceinline__ float I2F(int intVal) {
  return __int_as_float((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

__device__ float atomicMin_float(float *address, float val) {
  int val_int = F2I(val);
  int old = atomicMin((int *)address, val_int);
  return I2F(old);
}

__device__ float atomicMax_float(float *address, float val) {
  int val_int = F2I(val);
  int old = atomicMax((int *)address, val_int);
  return I2F(old);
}

void check(cudaError_t status, const char *message) {
  if (status != cudaSuccess) cout << message << endl;
}

__global__ void Norm(float *point, float *norm, int size, int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
      float tmp = point[tid * dim + i];
      dist += tmp * tmp;
    }
    norm[tid] = dist;
  }
}

__global__ void AddAll(float *queryNorm_dev, float *repNorm_dev,
                       float *query2reps_dev, int size, int rep_nb) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;
  if (tx < size && ty < rep_nb) {
    float temp = query2reps_dev[ty * size + tx];
    temp += (queryNorm_dev[tx] + repNorm_dev[ty]);
    query2reps_dev[ty * size + tx] = sqrt(temp);
  }
}

__global__ void findQCluster(float *query2reps_dev, P2R *q2rep_dev, int size,
                             int rep_nb, float *maxquery_dev,
                             R2all_static_dev *req2q_static_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    float temp = FLT_MAX;
    int index = -1;
    for (int i = 0; i < rep_nb; i++) {
      float tmp = query2reps_dev[i * size + tid];
      if (temp > tmp) {
        index = i;
        temp = tmp;
      }
    }
    q2rep_dev[tid] = {index, temp};
    atomicAdd(&req2q_static_dev[index].npoints, 1);
    atomicMax_float(&maxquery_dev[index], temp);
  }
}

__global__ void findTCluster(float *target2reps_dev, P2R *t2rep_dev, int size,
                             int rep_nb, R2all_static_dev *rep2t_static_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    float temp = FLT_MAX;
    int index = -1;
    for (int i = 0; i < rep_nb; i++) {
      float tmp = target2reps_dev[i * size + tid];
      if (temp > tmp) {
        index = i;
        temp = tmp;
      }
    }
    t2rep_dev[tid] = {index, temp};
    atomicAdd(&rep2t_static_dev[index].npoints, 1);
  }
}

__global__ void fillQMembers(P2R *q2rep_dev, int size, int *repsID,
                             R2all_dyn_p *req2q_dyn_p_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    int repId = q2rep_dev[tid].repIndex;
    int memberId = atomicAdd(&repsID[repId], 1);
    req2q_dyn_p_dev[repId].memberID[memberId] = tid;
  }
}

__global__ void fillTMembers(P2R *t2rep_dev, int size, int *repsID,
                             R2all_dyn_p *req2s_dyn_p_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    int repId = t2rep_dev[tid].repIndex;
    int memberId = atomicAdd(&repsID[repId], 1);
    req2s_dyn_p_dev[repId].sortedmembers[memberId] = {tid,
                                                      t2rep_dev[tid].dist2rep};
  }
}

__device__ int reorder = 0;
__global__ void reorderMembers(int rep_nb, int *repsID, int *reorder_members,
                               R2all_dyn_p *req2q_dyn_p_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < rep_nb) {
    if (repsID[tid] != 0) {
      int reorderId = atomicAdd(&reorder, repsID[tid]);
      memcpy(reorder_members + reorderId, req2q_dyn_p_dev[tid].memberID,
             repsID[tid] * sizeof(int));
    }
  }
}

__global__ void selectReps_cuda(float *queries_dev, int n_queries,
                                float *qreps_dev, int qrep_nb, int *qIndex_dev,
                                int *totalSum_dev, int totalTest, int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < totalTest * qrep_nb * qrep_nb) {
    int test = tid / (qrep_nb * qrep_nb);
    int repId = int(tid % (qrep_nb * qrep_nb)) / qrep_nb;
    float distance = Edistance_128(
      queries_dev + qIndex_dev[test * qrep_nb + repId] * dim,
      queries_dev +
        qIndex_dev[test * qrep_nb + int(tid % (qrep_nb * qrep_nb)) % qrep_nb] *
          dim,
      dim);
    atomicAdd(&totalSum_dev[test], int(distance));
  }
}

__device__ int repTest = 0;
__global__ void selectReps_max(float *queries_dev, int n_queries,
                               float *qreps_dev, int qrep_nb, int *qIndex_dev,
                               int *totalSum_dev, int totalTest, int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    float distance = 0.0f;
    for (int i = 0; i < totalTest; i++) {
      if (distance < totalSum_dev[i]) {
        distance = totalSum_dev[i];
        repTest = i;
      }
    }
  }
}

__global__ void selectReps_copy(float *queries_dev, int n_queries,
                                float *qreps_dev, int qrep_nb, int *qIndex_dev,
                                int *totalSum_dev, int totalTest, int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < qrep_nb) {
    memcpy(qreps_dev + tid * dim,
           queries_dev + qIndex_dev[repTest * qrep_nb + tid] * dim,
           dim * sizeof(float));
  }
}

void clusterReps(float *&queries_dev, float *&targets_dev, float *&qreps_dev,
                 float *&treps_dev, float *&maxquery_dev, P2R *&q2rep_dev,
                 P2R *&t2rep_dev, R2all_static_dev *&rep2q_static_dev,
                 R2all_static_dev *&rep2t_static_dev,
                 R2all_dyn_p *&rep2q_dyn_p_dev, R2all_dyn_p *&rep2t_dyn_p_dev,
                 float *&query2reps_dev, R2all_static *&rep2q_static,
                 R2all_static *&rep2t_static, R2all_dyn_p *&rep2q_dyn_p,
                 R2all_dyn_p *&rep2t_dyn_p, int *&reorder_members, int dim,
                 int n_queries, int qrep_nb, int n_targets, int trep_nb,
                 float *queries, float *targets, int K) {
  struct timespec t1, t3, t4, t35;
  cublasHandle_t h;
  cublasStatus_t cstat;
  cstat = cublasCreate(&h);

  if (cstat != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS initialization failed\n");
  }

  float alpha = -2.0;
  float beta = 0.0;

  cudaError_t status;
  status = cudaMalloc((void **)&queries_dev, n_queries * dim * sizeof(float));
  check(status, "Malloc queries failed\n");
  status = cudaMemcpy(queries_dev, queries, n_queries * dim * sizeof(float),
                      cudaMemcpyHostToDevice);
  check(status, "Memcpy queries failed\n");

  status = cudaMalloc((void **)&targets_dev, n_targets * dim * sizeof(float));
  check(status, "Malloc targets failed\n");
  status = cudaMemcpy(targets_dev, targets, n_targets * dim * sizeof(float),
                      cudaMemcpyHostToDevice);
  check(status, "Mem targets failed\n");

  status = cudaMalloc((void **)&qreps_dev, qrep_nb * dim * sizeof(float));
  check(status, "Malloc qreps failed\n");

  status = cudaMalloc((void **)&treps_dev, trep_nb * dim * sizeof(float));
  check(status, "Malloc treps failed\n");

  int totalTest = 10;
  int *qIndex_dev, *qIndex;
  qIndex = (int *)malloc(totalTest * qrep_nb * sizeof(int));
  cudaMalloc((void **)&qIndex_dev, qrep_nb * totalTest * sizeof(int));
  srand(2015);
  for (int i = 0; i < totalTest; i++) {
    for (int j = 0; j < qrep_nb; j++) {
      qIndex[i * qrep_nb + j] = rand() % n_queries;
    }
  }

  cudaMemcpy(qIndex_dev, qIndex, totalTest * qrep_nb * sizeof(int),
             cudaMemcpyHostToDevice);
  int *totalSum_dev;
  cudaMalloc((void **)&totalSum_dev, totalTest * sizeof(float));
  cudaMemset(totalSum_dev, 0, totalTest * sizeof(float));

  selectReps_cuda<<<(totalTest * qrep_nb * qrep_nb + 255) / 256, 256>>>(
    queries_dev, n_queries, qreps_dev, qrep_nb, qIndex_dev, totalSum_dev,
    totalTest, dim);

  cudaDeviceSynchronize();
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    printf("cudaGetLastError() returned %d: %s\n", status,
           cudaGetErrorString(status));
  }

  selectReps_max<<<1, 1>>>(queries_dev, n_queries, qreps_dev, qrep_nb,
                           qIndex_dev, totalSum_dev, totalTest, dim);
  selectReps_copy<<<(qrep_nb + 255) / 256, 256>>>(
    queries_dev, n_queries, qreps_dev, qrep_nb, qIndex_dev, totalSum_dev,
    totalTest, dim);

  int *sIndex_dev, *sIndex;

  sIndex = (int *)malloc(totalTest * trep_nb * sizeof(int));
  cudaMalloc((void **)&sIndex_dev, trep_nb * totalTest * sizeof(int));

  srand(2015);
  for (int i = 0; i < totalTest; i++) {
    for (int j = 0; j < trep_nb; j++) {
      sIndex[i * trep_nb + j] = rand() % n_targets;
    }
  }
  cudaMemcpy(sIndex_dev, sIndex, totalTest * trep_nb * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemset(totalSum_dev, 0, totalTest * sizeof(float));

  selectReps_cuda<<<(totalTest * trep_nb * trep_nb + 255) / 256, 256>>>(
    targets_dev, n_targets, treps_dev, trep_nb, sIndex_dev, totalSum_dev,
    totalTest, dim);
  selectReps_max<<<1, 1>>>(targets_dev, n_targets, treps_dev, trep_nb,
                           sIndex_dev, totalSum_dev, totalTest, dim);

  selectReps_copy<<<(trep_nb + 255) / 256, 256>>>(
    targets_dev, n_targets, treps_dev, trep_nb, sIndex_dev, totalSum_dev,
    totalTest, dim);

  cudaDeviceSynchronize();

  cudaMalloc((void **)&rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev));
  cudaMalloc((void **)&rep2t_static_dev, trep_nb * sizeof(R2all_static_dev));

  float *queryNorm_dev, *qrepNorm_dev, *targetNorm_dev, *srepNorm_dev;
  cudaMalloc((void **)&queryNorm_dev, n_queries * sizeof(float));
  cudaMalloc((void **)&targetNorm_dev, n_targets * sizeof(float));
  cudaMalloc((void **)&qrepNorm_dev, qrep_nb * sizeof(float));
  cudaMalloc((void **)&srepNorm_dev, trep_nb * sizeof(float));

  timePoint(t3);
  cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_queries, qrep_nb, dim, &alpha,
              queries_dev, dim, qreps_dev, dim, &beta, query2reps_dev,
              n_queries);
  cudaDeviceSynchronize();
  timePoint(t35);
  timePoint(t1);
  Norm<<<(n_queries + 255) / 256, 256>>>(queries_dev, queryNorm_dev, n_queries,
                                         dim);

  cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_queries, qrep_nb, dim, &alpha,
              queries_dev, dim, qreps_dev, dim, &beta, query2reps_dev,
              n_queries);

  cudaDeviceSynchronize();
  timePoint(t3);
  Norm<<<(qrep_nb + 255) / 256, 256>>>(qreps_dev, qrepNorm_dev, qrep_nb, dim);
  dim3 block2D(16, 16, 1);
  dim3 grid2D_q((n_queries + 15) / 16, (qrep_nb + 15) / 16, 1);
  AddAll<<<grid2D_q, block2D>>>(queryNorm_dev, qrepNorm_dev, query2reps_dev,
                                n_queries, qrep_nb);

  status = cudaMalloc((void **)&maxquery_dev, qrep_nb * sizeof(float));
  check(status, "Malloc maxquery_dev failed\n");
  cudaMemset(maxquery_dev, 0, qrep_nb * sizeof(float));

  status = cudaMalloc((void **)&q2rep_dev, n_queries * sizeof(P2R));
  check(status, "Malloc q2rep_dev failed\n");
  findQCluster<<<(n_queries + 255) / 256, 256>>>(
    query2reps_dev, q2rep_dev, n_queries, qrep_nb, maxquery_dev,
    rep2q_static_dev);

  int *qrepsID;
  cudaMalloc((void **)&qrepsID, qrep_nb * sizeof(int));
  cudaMemset(qrepsID, 0, qrep_nb * sizeof(int));
  status =
    cudaMemcpy(rep2q_static, rep2q_static_dev,
               qrep_nb * sizeof(R2all_static_dev), cudaMemcpyDeviceToHost);
  check(status, "Memcpy rep2q_static failed\n");
  for (int i = 0; i < qrep_nb; i++) {
    cudaMalloc((void **)&rep2q_dyn_p[i].replist, trep_nb * sizeof(IndexDist));
    cudaMalloc((void **)&rep2q_dyn_p[i].kubound, K * sizeof(float));
    cudaMalloc((void **)&rep2q_dyn_p[i].memberID,
               rep2q_static[i].npoints * sizeof(int));
  }

  cudaMalloc((void **)&rep2q_dyn_p_dev, qrep_nb * sizeof(R2all_dyn_p));
  cudaMemcpy(rep2q_dyn_p_dev, rep2q_dyn_p, qrep_nb * sizeof(R2all_dyn_p),
             cudaMemcpyHostToDevice);
  fillQMembers<<<(n_queries + 255) / 256, 256>>>(q2rep_dev, n_queries, qrepsID,
                                                 rep2q_dyn_p_dev);

  status = cudaMalloc((void **)&reorder_members, n_queries * sizeof(int));
  check(status, "Malloc reorder_members failed\n");

  reorderMembers<<<(qrep_nb + 255) / 256, 256>>>(
    qrep_nb, qrepsID, reorder_members, rep2q_dyn_p_dev);

  cudaDeviceSynchronize();
  float *target2reps = (float *)malloc(n_targets * trep_nb * sizeof(float));
  float *target2reps_dev;
  status =
    cudaMalloc((void **)&target2reps_dev, n_targets * trep_nb * sizeof(float));
  check(status, "Malloc target2reps_dev failed\n");

  cudaDeviceSynchronize();
  Norm<<<(n_targets + 255) / 256, 256>>>(targets_dev, targetNorm_dev, n_targets,
                                         dim);
  cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_targets, trep_nb, dim, &alpha,
              targets_dev, dim, treps_dev, dim, &beta, target2reps_dev,
              n_targets);
  cudaDeviceSynchronize();
  Norm<<<(trep_nb + 255) / 256, 256>>>(treps_dev, srepNorm_dev, trep_nb, dim);
  dim3 grid2D_s((n_targets + 15) / 16, (trep_nb + 15) / 16, 1);
  AddAll<<<grid2D_s, block2D>>>(targetNorm_dev, srepNorm_dev, target2reps_dev,
                                n_targets, trep_nb);

  status = cudaMalloc((void **)&t2rep_dev, n_targets * sizeof(P2R));
  check(status, "Malloc t2rep_dev failed\n");
  findTCluster<<<(n_targets + 255) / 256, 256>>>(
    target2reps_dev, t2rep_dev, n_targets, trep_nb, rep2t_static_dev);
  int *srepsID;
  cudaMalloc((void **)&srepsID, trep_nb * sizeof(int));
  cudaMemset(srepsID, 0, trep_nb * sizeof(int));
  cudaMemcpy(rep2t_static, rep2t_static_dev, trep_nb * sizeof(R2all_static_dev),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < trep_nb; i++) {
    cudaMalloc((void **)&rep2t_dyn_p[i].sortedmembers,
               rep2t_static[i].npoints * sizeof(R2all_dyn_p));
  }
  cudaMalloc((void **)&rep2t_dyn_p_dev, trep_nb * sizeof(R2all_dyn_p));
  cudaMemcpy(rep2t_dyn_p_dev, rep2t_dyn_p, trep_nb * sizeof(R2all_dyn_p),
             cudaMemcpyHostToDevice);
  fillTMembers<<<(n_targets + 255) / 256, 256>>>(t2rep_dev, n_targets, srepsID,
                                                 rep2t_dyn_p_dev);
  timePoint(t3);

#pragma omp parallel for
  for (int i = 0; i < trep_nb; i++) {
    if (rep2t_static[i].npoints > 0) {
      vector<IndexDist> temp;
      temp.resize(rep2t_static[i].npoints);
      cudaMemcpy(&temp[0], rep2t_dyn_p[i].sortedmembers,
                 rep2t_static[i].npoints * sizeof(IndexDist),
                 cudaMemcpyDeviceToHost);
      sort(temp.begin(), temp.end(), sort_inc());
      cudaMemcpy(rep2t_dyn_p[i].sortedmembers, &temp[0],
                 rep2t_static[i].npoints * sizeof(IndexDist),
                 cudaMemcpyHostToDevice);
    }
  }

  free(qIndex);
  cudaFree(qIndex_dev);
  free(sIndex);
  cudaFree(sIndex_dev);
  timePoint(t4);
  cudaFree(query2reps_dev);
  cudaMalloc((void **)&query2reps_dev, n_queries * trep_nb * sizeof(float));
  dim3 grid2D_qsrep((n_queries + 15) / 16, (trep_nb + 15) / 16, 1);
  cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_queries, trep_nb, dim, &alpha,
              queries_dev, dim, treps_dev, dim, &beta, query2reps_dev,
              n_queries);
  AddAll<<<grid2D_qsrep, block2D>>>(queryNorm_dev, srepNorm_dev, query2reps_dev,
                                    n_queries, trep_nb);
  cublasDestroy(h);
}

__global__ void RepsUpperBound(float *qreps_dev, float *treps_dev,
                               float *maxquery_dev,
                               R2all_static_dev *rep2q_static_dev,
                               R2all_dyn_p *rep2q_dyn_p_dev,
                               R2all_static_dev *rep2t_static_dev,
                               R2all_dyn_p *rep2t_dyn_p_dev, int qrep_nb,
                               int trep_nb, int dim, int K) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < qrep_nb) {
    int UBoundCount = 0;
    for (int i = 0; i < trep_nb; i++) {
      float rep2rep =
        Edistance_128(qreps_dev + tid * dim, treps_dev + i * dim, dim);
      int count = 0;
      while (count < K && count < rep2t_static_dev[i].npoints) {
        float g2pUBound = maxquery_dev[tid] + rep2rep +
                          rep2t_dyn_p_dev[i].sortedmembers[count].dist;

        if (UBoundCount < K) {
          rep2q_dyn_p_dev[tid].kubound[UBoundCount] = g2pUBound;
          if (rep2q_static_dev[tid].kuboundMax < g2pUBound)
            rep2q_static_dev[tid].kuboundMax = g2pUBound;

          UBoundCount++;
        } else {
          if (rep2q_static_dev[tid].kuboundMax > g2pUBound) {
            float max_local = 0.0f;
            for (int j = 0; j < K; j++) {
              if (rep2q_dyn_p_dev[tid].kubound[j] ==
                  rep2q_static_dev[tid].kuboundMax) {
                rep2q_dyn_p_dev[tid].kubound[j] = g2pUBound;
              }
              if (max_local < rep2q_dyn_p_dev[tid].kubound[j]) {
                max_local = rep2q_dyn_p_dev[tid].kubound[j];
              }
            }
            rep2q_static_dev[tid].kuboundMax = max_local;
          }
        }
        count++;
      }
    }
  }
}

__global__ void FilterReps(float *qreps_dev, float *treps_dev,
                           float *maxquery_dev,
                           R2all_static_dev *rep2q_static_dev,
                           R2all_dyn_p *rep2q_dyn_p_dev,
                           R2all_static_dev *rep2t_static_dev, int qrep_nb,
                           int trep_nb, int dim) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy =
    threadIdx.y + blockIdx.y * blockDim.y;  // calculate reps[tidy].replist;
  if (tidx < trep_nb && tidy < qrep_nb) {
    float distance =
      Edistance(qreps_dev + tidy * dim, treps_dev + tidx * dim, dim);
    if (distance - maxquery_dev[tidy] - rep2t_static_dev[tidx].maxdist <
        rep2q_static_dev[tidy].kuboundMax) {
      int rep_id = atomicAdd(&rep2q_static_dev[tidy].noreps, 1);
      rep2q_dyn_p_dev[tidy].replist[rep_id].index = tidx;
      rep2q_dyn_p_dev[tidy].replist[rep_id].dist = distance;
    }
  }
}

__device__ int Total = 0;
__global__ void printTotal() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
#ifdef debug
    printf("Total %d\n", Total);
#endif
  }
}

__global__ void KNNQuery_base(
  float *queries_dev, float *targets_dev, float *query2reps_dev, P2R *q2rep_dev,
  R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,
  R2all_static_dev *rep2t_static_dev, R2all_dyn_p *rep2t_dyn_p_dev,
  int n_queries, int dim, int K, IndexDist *knearest1, int *reorder_members) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_queries) {
    tid = reorder_members[tid];
    int repIndex = q2rep_dev[tid].repIndex;
    float theta = rep2q_static_dev[repIndex].kuboundMax;
    int Kcount = 0;
    int count = 0;

    IndexDist knearest[1000];
    for (int i = 0; i < rep2q_static_dev[repIndex].noreps; i++) {
      int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
      float query2rep = 0.0f;
      query2rep = query2reps_dev[tid + minlb_rid * n_queries];

      for (int j = rep2t_static_dev[minlb_rid].npoints - 1; j >= 0; j--) {
        IndexDist targetj = rep2t_dyn_p_dev[minlb_rid].sortedmembers[j];
        float p2plbound = query2rep - targetj.dist;
        if (p2plbound > theta)
          break;
        else if (p2plbound < theta * (-1.0f))
          continue;
        else if (p2plbound <= theta && p2plbound >= theta * (-1.0f)) {
          float query2target = Edistance_128(
            queries_dev + tid * dim, targets_dev + targetj.index * dim, dim);
          count++;
          atomicAdd(&Total, 1);
          int insert = -1;
          for (int kk = 0; kk < Kcount; kk++) {
            if (query2target < knearest[kk].dist) {
              insert = kk;
              break;
            }
          }
          if (Kcount < K) {
            if (insert == -1) {
              knearest[Kcount] = {targetj.index, query2target};
            } else {
              for (int move = Kcount - 1; move >= insert; move--) {
                knearest[move + 1] = knearest[move];
              }
              knearest[insert] = {targetj.index, query2target};
            }
            Kcount++;
          } else {  // Kcount = K
            if (insert == -1) {
              continue;
            } else {
              for (int move = K - 2; move >= insert; move--) {
                knearest[move + 1] = knearest[move];
              }

              knearest[insert] = {targetj.index, query2target};
              theta = knearest[K - 1].dist;
            }
          }
        }
      }
    }
    memcpy(&knearest1[tid * K], knearest, K * sizeof(IndexDist));
  }
}

__global__ void KNNQuery_theta(P2R *q2rep_dev,
                               R2all_static_dev *rep2q_static_dev,
                               int n_queries, float *thetas) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_queries) {
    int repIndex = q2rep_dev[tid].repIndex;
    thetas[tid] = rep2q_static_dev[repIndex].kuboundMax;
  }
}

__global__ void KNNQuery(
  float *queries_dev, float *targets_dev, float *qreps_dev, float *treps_dev,
  float *query2reps_dev, float *maxquery_dev, P2R *q2rep_dev, P2R *t2rep_dev,
  R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,
  R2all_static_dev *rep2t_static_dev, R2all_dyn_p *rep2t_dyn_p_dev,
  int n_queries, int n_targets, int qrep_nb, int trep_nb, int dim, int K,
  IndexDist *knearest, float *thetas, int tpq, int *reorder_members) {
  int ttid = threadIdx.x + blockIdx.x * blockDim.x;
  int tp = ttid % tpq;
  int tid = ttid / tpq;
  if (tid < n_queries) {
    tid = reorder_members[tid];
    ttid = tid * tpq + tp;
    int repIndex = q2rep_dev[tid].repIndex;
    int Kcount = 0;
    int count = 0;

    for (int i = 0; i < rep2q_static_dev[repIndex].noreps; i++) {
      int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
      float query2rep = 0.0f;
      query2rep = Edistance_128(queries_dev + tid * dim,
                                treps_dev + minlb_rid * dim, dim);

      for (int j = rep2t_static_dev[minlb_rid].npoints - 1 - tp; j >= 0;
           j -= tpq) {
        IndexDist targetj = rep2t_dyn_p_dev[minlb_rid].sortedmembers[j];
        float p2plbound = query2rep - targetj.dist;
        if (p2plbound > *(volatile float *)&thetas[tid]) {
          break;
        } else if (p2plbound < *(volatile float *)&thetas[tid] * (-1.0f)) {
          continue;
        } else if (p2plbound <= *(volatile float *)&thetas[tid] &&
                   p2plbound >= *(volatile float *)&thetas[tid] * (-1.0f)) {
          float query2target = Edistance_128(
            queries_dev + tid * dim, targets_dev + targetj.index * dim, dim);
          count++;
          atomicAdd(&Total, 1);
          int insert = -1;
          for (int kk = 0; kk < Kcount; kk++) {
            if (query2target < knearest[ttid * K + kk].dist) {
              insert = kk;
              break;
            }
          }
          if (Kcount < K) {
            if (insert == -1) {
              knearest[ttid * K + Kcount] = {targetj.index, query2target};
            } else {
              for (int move = Kcount - 1; move >= insert; move--) {
                knearest[ttid * K + move + 1] = knearest[ttid * K + move];
              }
              knearest[ttid * K + insert] = {targetj.index, query2target};
            }
            Kcount++;
          } else {  // Kcount = K
            if (insert == -1) {
              continue;
            } else {
              for (int move = K - 2; move >= insert; move--) {
                knearest[ttid * K + move + 1] = knearest[ttid * K + move];
              }

              knearest[ttid * K + insert] = {targetj.index, query2target};
              atomicMin_float(&thetas[tid], knearest[ttid * K + K - 1].dist);
            }
          }
        }
      }
    }
  }
}

__global__ void final(int k, IndexDist *knearest, int tpq, int n_queries,
                      IndexDist *final_knearest, int *tag_base) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int *tag = tid * tpq + tag_base;
  if (tid < n_queries) {
    for (int i = 0; i < k; i++) {
      float min = knearest[tid * tpq * k + tag[0]].dist;
      int index = 0;
      for (int j = 1; j < tpq; j++) {
        float value = knearest[(tid * tpq + j) * k + tag[j]].dist;
        if (min > value) {
          min = value;
          index = j;
        }
      }

      final_knearest[tid * k + i] =
        knearest[(tid * tpq + index) * k + tag[index]];
      tag[index]++;
    }
  }
}

void sweet_knn(float *query_points, int n_queries, int D, float *target_points,
               int n_targets, int64_t *res_I, float *res_D, int k, int qrep_nb,
               int trep_nb) {
  struct timespec t1, t2;
  timePoint(t1);

  R2all_static *rep2q_static =
    (R2all_static *)malloc(qrep_nb * sizeof(R2all_static));
  R2all_static *rep2t_static =
    (R2all_static *)malloc(trep_nb * sizeof(R2all_static));

  float *queries_dev, *targets_dev, *qreps_dev, *treps_dev;
  P2R *q2rep_dev, *t2rep_dev;
  R2all_static_dev *rep2q_static_dev;
  R2all_dyn_p *rep2q_dyn_p_dev;
  R2all_static_dev *rep2t_static_dev;
  R2all_dyn_p *rep2t_dyn_p_dev;
  float *query2reps_dev;
  float *maxquery_dev;
  int *reorder_members;

  R2all_dyn_p *rep2q_dyn_p =
    (R2all_dyn_p *)malloc(qrep_nb * sizeof(R2all_dyn_p));
  R2all_dyn_p *rep2t_dyn_p =
    (R2all_dyn_p *)malloc(trep_nb * sizeof(R2all_dyn_p));
  // Select reps
  timePoint(t1);
  // cluster queries and targets to reps
  cudaMalloc((void **)&query2reps_dev, qrep_nb * n_queries * sizeof(float));

  clusterReps(queries_dev, targets_dev, qreps_dev, treps_dev, maxquery_dev,
              q2rep_dev, t2rep_dev, rep2q_static_dev, rep2t_static_dev,
              rep2q_dyn_p_dev, rep2t_dyn_p_dev, query2reps_dev, rep2q_static,
              rep2t_static, rep2q_dyn_p, rep2t_dyn_p, reorder_members, D,
              n_queries, qrep_nb, n_targets, trep_nb, query_points,
              target_points, k);

  if (cudaGetLastError() != cudaSuccess) {
    cout << "error 16" << endl;
  }

  // Kernel 1: upperbound for each rep
  RepsUpperBound<<<(qrep_nb + 255) / 256, 256>>>(
    qreps_dev, treps_dev, maxquery_dev, rep2q_static_dev, rep2q_dyn_p_dev,
    rep2t_static_dev, rep2t_dyn_p_dev, qrep_nb, trep_nb, D, k);

  if (cudaGetLastError() != cudaSuccess) {
    cout << "Kernel RepsUpperBound failed" << endl;
  }

  // Kernel 2: filter reps	based on upperbound and lowerbound;
  dim3 block(16, 16, 1);
  dim3 grid((trep_nb + block.x - 1) / block.x,
            (qrep_nb + block.y - 1) / block.y, 1);
  FilterReps<<<grid, block>>>(qreps_dev, treps_dev, maxquery_dev,
                              rep2q_static_dev, rep2q_dyn_p_dev,
                              rep2t_static_dev, qrep_nb, trep_nb, D);

  cudaMemcpy(rep2q_static, rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev),
             cudaMemcpyDeviceToHost);

#pragma omp parallel for
  for (int i = 0; i < qrep_nb; i++) {
    vector<IndexDist> temp;
    temp.resize(rep2q_static[i].noreps);
    cudaMemcpy(&temp[0], rep2q_dyn_p[i].replist,
               rep2q_static[i].noreps * sizeof(IndexDist),
               cudaMemcpyDeviceToHost);
    sort(temp.begin(), temp.end(), sort_inc());
    cudaMemcpy(rep2q_dyn_p[i].replist, &temp[0],
               rep2q_static[i].noreps * sizeof(IndexDist),
               cudaMemcpyHostToDevice);
  }

  // Kernel 3: knn for each point
  IndexDist *knearest, *final_knearest;
  int tpq = (2048 * 13) / n_queries;
  IndexDist *knearest_h =
    (IndexDist *)malloc(n_queries * k * sizeof(IndexDist));
  cudaMalloc((void **)&knearest, n_queries * (tpq + 1) * k * sizeof(IndexDist));
  if (tpq > 1) {
    float *theta;
    cudaMalloc((void **)&theta, n_queries * sizeof(float));
    KNNQuery_theta<<<(n_queries + 255) / 256, 256>>>(
      q2rep_dev, rep2q_static_dev, n_queries, theta);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    KNNQuery<<<(tpq * n_queries + 255) / 256, 256>>>(
      queries_dev, targets_dev, qreps_dev, treps_dev, query2reps_dev,
      maxquery_dev, q2rep_dev, t2rep_dev, rep2q_static_dev, rep2q_dyn_p_dev,
      rep2t_static_dev, rep2t_dyn_p_dev, n_queries, n_targets, qrep_nb, trep_nb,
      D, k, knearest, theta, tpq, reorder_members);
    final_knearest = knearest + n_queries * tpq * k;

    int *tag_base;
    cudaMalloc((void **)&tag_base, tpq * n_queries * sizeof(int));
    cudaMemset(tag_base, 0, tpq * n_queries * sizeof(int));
    final<<<(n_queries + 255) / 256, 256>>>(k, knearest, tpq, n_queries,
                                            final_knearest, tag_base);
    cudaFree(theta);
    cudaFree(tag_base);
  } else {
    KNNQuery_base<<<(n_queries + 255) / 256, 256>>>(
      queries_dev, targets_dev, query2reps_dev, q2rep_dev, rep2q_static_dev,
      rep2q_dyn_p_dev, rep2t_static_dev, rep2t_dyn_p_dev, n_queries, D, k,
      knearest, reorder_members);
  }

  cudaDeviceSynchronize();
  timePoint(t2);
  printf("total time %f\n", timeLen(t1, t2));
  printTotal<<<1, 1>>>();
  if (tpq > 1) {
    cudaMemcpy(knearest_h, final_knearest, n_queries * k * sizeof(IndexDist),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(knearest_h, knearest, n_queries * k * sizeof(IndexDist),
               cudaMemcpyDeviceToHost);
  }

  cudaFree(knearest);
  cudaDeviceSynchronize();

  for (int i = 0; i < n_queries; i++) {
    for (int j = 0; j < k; j++) {
      res_I[i * k + j] = knearest_h[i * k + j].index;
      res_D[i * k + j] = knearest_h[i * k + j].dist;
    }
  }

  free(knearest_h);
  free(rep2q_static);
  free(rep2t_static);
  free(rep2q_dyn_p);
  free(rep2t_dyn_p);
}
// END SWEET KNN CODE

void knn_classify(cumlHandle &handle, int *out, int64_t *knn_indices,
                  std::vector<int *> &y, size_t n_samples, int k) {
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  std::vector<int *> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (int i = 0; i < y.size(); i++) {
    MLCommon::Label::getUniqueLabels(y[i], n_samples, &(uniq_labels[i]),
                                     &(n_unique[i]), stream, d_alloc);
  }

  MLCommon::Selection::knn_classify(out, knn_indices, y, n_samples, k,
                                    uniq_labels, n_unique, d_alloc, stream);
}

void knn_regress(cumlHandle &handle, float *out, int64_t *knn_indices,
                 std::vector<float *> &y, size_t n_samples, int k) {
  MLCommon::Selection::knn_regress(out, knn_indices, y, n_samples, k,
                                   handle.getStream());
}

void knn_class_proba(cumlHandle &handle, std::vector<float *> &out,
                     int64_t *knn_indices, std::vector<int *> &y,
                     size_t n_samples, int k) {
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  std::vector<int *> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (int i = 0; i < y.size(); i++) {
    MLCommon::Label::getUniqueLabels(y[i], n_samples, &(uniq_labels[i]),
                                     &(n_unique[i]), stream, d_alloc);
  }

  MLCommon::Selection::class_probs(out, knn_indices, y, n_samples, k,
                                   uniq_labels, n_unique, d_alloc, stream);
}

/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
kNN::kNN(const cumlHandle &handle, int D, bool verbose)
  : D(D), total_n(0), indices(0), verbose(verbose) {
  this->handle = const_cast<cumlHandle *>(&handle);
  sizes = nullptr;
  ptrs = nullptr;
}

kNN::~kNN() {
  if (this->indices > 0) {
    reset();
  }
}

void kNN::reset() {
  if (this->indices > 0) {
    this->indices = 0;
    this->total_n = 0;

    delete[] this->ptrs;
    delete[] this->sizes;
  }
}

/**
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 * @param rowMajor is the input in rowMajor?
	 */
void kNN::fit(std::vector<float *> &input, std::vector<int> &sizes,
              bool rowMajor) {
  this->rowMajorIndex = rowMajor;

  int N = input.size();

  if (this->verbose) std::cout << "N=" << N << std::endl;

  reset();

  this->indices = N;
  this->ptrs = (float **)malloc(N * sizeof(float *));
  this->sizes = (int *)malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    this->ptrs[i] = input[i];
    this->sizes[i] = sizes[i];
  }
}

/**
	 * Search the kNN for the k-nearest neighbors of a set of query vectors
	 * @param search_items set of vectors to query for neighbors
	 * @param n 		   number of items in search_items
	 * @param res_I 	   pointer to device memory for returning k nearest indices
	 * @param res_D		   pointer to device memory for returning k nearest distances
	 * @param k			   number of neighbors to query
	 * @param rowMajor is the query array in row major layout?
	 */
void kNN::search(float *search_items, int n, int64_t *res_I, float *res_D,
                 int k, bool rowMajor) {
  ASSERT(this->indices > 0, "Cannot search before model has been trained.");

  std::vector<cudaStream_t> int_streams =
    handle->getImpl().getInternalStreams();

  MLCommon::Selection::brute_force_knn(
    ptrs, sizes, indices, D, search_items, n, res_I, res_D, k,
    handle->getImpl().getDeviceAllocator(), handle->getImpl().getStream(),
    int_streams.data(), handle->getImpl().getNumInternalStreams(),
    this->rowMajorIndex, rowMajor);
}
};  // namespace ML

/**
 * @brief Flat C API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param handle the cuml handle to use
 * @param input an array of pointers to the input arrays
 * @param sizes an array of sizes of input arrays
 * @param n_params array size of input and sizes
 * @param D the dimensionality of the arrays
 * @param search_items array of items to search of dimensionality D
 * @param n number of rows in search_items
 * @param res_I the resulting index array of size n * k
 * @param res_D the resulting distance array of size n * k
 * @param k the number of nearest neighbors to return
 * @param rowMajorIndex is the index array in row major layout?
 * @param rowMajorQuery is the query array in row major layout?
 */
extern "C" cumlError_t knn_search(const cumlHandle_t handle, float **input,
                                  int *sizes, int n_params, int D,
                                  float *search_items, int n, int64_t *res_I,
                                  float *res_D, int k, bool rowMajorIndex,
                                  bool rowMajorQuery) {
  cumlError_t status;

  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);

  std::vector<cudaStream_t> int_streams =
    handle_ptr->getImpl().getInternalStreams();

  std::vector<float *> input_vec(n_params);
  std::vector<int> sizes_vec(n_params);
  for (int i = 0; i < n_params; i++) {
    input_vec.push_back(input[i]);
    sizes_vec.push_back(sizes[i]);
  }

  if (status == CUML_SUCCESS) {
    try {
      MLCommon::Selection::brute_force_knn(
        input_vec, sizes_vec, D, search_items, n, res_I, res_D, k,
        handle_ptr->getImpl().getDeviceAllocator(),
        handle_ptr->getImpl().getStream(), int_streams.data(),
        handle_ptr->getImpl().getNumInternalStreams(), rowMajorIndex,
        rowMajorQuery);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
