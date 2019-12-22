#include "cublas_v2.h"
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <vector>

using namespace std;

void pointSetup(char *query_data, int n_queries, float *queries,
                char *target_data, int n_targets, float *targets, int dim) {

  srand(2015);
  if (strcmp((const char *)query_data, "random") == 0) {
    cout << "Query data random" << endl;
    for (int i = 0; i < n_queries; i++) {
      for (int j = 0; j < dim; j++) {
        queries[i * dim + j] = rand() % 10 + (float)rand() / RAND_MAX;
      }
    }
  } else {
    FILE *fq = fopen(query_data, "r");
    if (fq == NULL) {
      cout << "Error opening query file" << endl;
      exit(1);
    }
    for (int i = 0; i < n_queries; i++) {
      for (int j = 0; j < dim; j++) {
        if (fscanf(fq, "%f", &queries[i * dim + j]) != 1) {
          printf("Error reading query file\n");
        }
      }
    }
    fclose(fq);
  }

  if (strcmp((const char *)target_data, "random") == 0) {
    cout << "Source data random" << endl;
    for (int i = 0; i < n_targets; i++) {
      for (int j = 0; j < dim; j++) {
        targets[i * dim + j] = rand() % 10 + (float)rand() / RAND_MAX;
      }
    }
  } else {
    FILE *fs = fopen(target_data, "r");
    if (fs == NULL) {
      cout << "Error opening target file" << endl;
      exit(1);
    }
    for (int i = 0; i < n_targets; i++) {
      for (int j = 0; j < dim; j++) {
        if (fscanf(fs, "%f", &targets[i * dim + j]) != 1) {
          printf("Error reading target file\n");
        }
      }
    }
    fclose(fs);
  }
}

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
  if (status != cudaSuccess)
    cout << message << endl;
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
        queries_dev + qIndex_dev[test * qrep_nb +
                                 int(tid % (qrep_nb * qrep_nb)) % qrep_nb] *
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
    printf("repTest %d\n", repTest);
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
  printf("cublasSgemm warm up time %f\n", timeLen(t3, t35));
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

  timePoint(t35);
  printf("query rep first part time %f\n", timeLen(t3, t35));

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
  timePoint(t4);
  printf("query rep time  %f\n", timeLen(t3, t4));
  float *target2reps = (float *)malloc(n_targets * trep_nb * sizeof(float));
  float *target2reps_dev;
  status = cudaMalloc((void **)&target2reps_dev,
                      n_targets * trep_nb * sizeof(float));
  check(status, "Malloc target2reps_dev failed\n");

  cudaDeviceSynchronize();
  timePoint(t3);
  Norm<<<(n_targets + 255) / 256, 256>>>(targets_dev, targetNorm_dev, n_targets,
                                         dim);
  cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_targets, trep_nb, dim, &alpha,
              targets_dev, dim, treps_dev, dim, &beta, target2reps_dev,
              n_targets);
  cudaDeviceSynchronize();
  timePoint(t35);
  printf("target rep first part time %f\n", timeLen(t3, t35));
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
  printf("target rep time %f\n", timeLen(t3, t4));
  cublasDestroy(h);
}

__global__ void
RepsUpperBound(float *qreps_dev, float *treps_dev, float *maxquery_dev,
               R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,
               R2all_static_dev *rep2t_static_dev, R2all_dyn_p *rep2t_dyn_p_dev,
               int qrep_nb, int trep_nb, int dim, int K) {
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
      threadIdx.y + blockIdx.y * blockDim.y; // calculate reps[tidy].replist;
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
    printf("Total %d\n", Total);
  }
}

__global__ void
KNNQuery_base(float *queries_dev, float *targets_dev, float *query2reps_dev,
              P2R *q2rep_dev, R2all_static_dev *rep2q_static_dev,
              R2all_dyn_p *rep2q_dyn_p_dev, R2all_static_dev *rep2t_static_dev,
              R2all_dyn_p *rep2t_dyn_p_dev, int n_queries, int dim, int K,
              IndexDist *knearest1, int *reorder_members) {
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
          } else { // Kcount = K
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

__global__ void
KNNQuery(float *queries_dev, float *targets_dev, float *qreps_dev,
         float *treps_dev, float *query2reps_dev, float *maxquery_dev,
         P2R *q2rep_dev, P2R *t2rep_dev, R2all_static_dev *rep2q_static_dev,
         R2all_dyn_p *rep2q_dyn_p_dev, R2all_static_dev *rep2t_static_dev,
         R2all_dyn_p *rep2t_dyn_p_dev, int n_queries, int n_targets,
         int qrep_nb, int trep_nb, int dim, int K, IndexDist *knearest,
         float *thetas, int tpq, int *reorder_members) {
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
          } else { // Kcount = K
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
  timePoint(t2);
  printf("cudaMalloc time %f\n", timeLen(t1, t2));

  clusterReps(queries_dev, targets_dev, qreps_dev, treps_dev, maxquery_dev,
              q2rep_dev, t2rep_dev, rep2q_static_dev, rep2t_static_dev,
              rep2q_dyn_p_dev, rep2t_dyn_p_dev, query2reps_dev, rep2q_static,
              rep2t_static, rep2q_dyn_p, rep2t_dyn_p, reorder_members, D,
              n_queries, qrep_nb, n_targets, trep_nb, query_points,
              target_points, k);

  timePoint(t2);
  printf("prep time %f\n", timeLen(t1, t2));
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

  struct timespec sort_start, sort_end;
  timePoint(sort_start);
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

  timePoint(sort_end);
  printf("sort query replist time %f\n", timeLen(sort_start, sort_end));

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
        rep2t_static_dev, rep2t_dyn_p_dev, n_queries, n_targets, qrep_nb,
        trep_nb, D, k, knearest, theta, tpq, reorder_members);
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

int main(int argc, char *argv[]) {

  if (argc < 10) {
    cout << "Usage: ./knnjoin query_nb target_nb dimension qrep_nb trep_nb k "
            "query_data target_data point_to_display\n";
    exit(1);
  }

  int n_queries = atoi(argv[1]);
  int n_targets = atoi(argv[2]);
  int dim = atoi(argv[3]);
  int qrep_nb = atoi(argv[4]);
  int trep_nb = atoi(argv[5]);
  int K = atoi(argv[6]);
  char *query_data = argv[7];
  char *target_data = argv[8];
  int i = atoi(argv[9]);

  float *target_points = (float *)malloc(n_targets * dim * sizeof(float));
  float *query_points = (float *)malloc(n_queries * dim * sizeof(float));

  // Setup for target and query points.
  pointSetup(query_data, n_queries, query_points, target_data, n_targets,
             target_points, dim);

  int64_t *res_I = (int64_t *)malloc(n_queries * K * sizeof(int64_t));
  float *res_D = (float *)malloc(n_queries * K * sizeof(float));

  sweet_knn(query_points, n_queries, dim, target_points, n_targets, res_I,
            res_D, K, qrep_nb, trep_nb);

  printf("Displaying results for index %d\n", i);
  for (int j = 0; j < K; j++) {
    printf("%d:\t%ld\t%f\n", j, res_I[i * K + j], res_D[i * K + j]);
  }

  free(query_points);
  free(target_points);
  free(res_I);
  free(res_D);

  return 0;
}
