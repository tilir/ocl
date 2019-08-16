//------------------------------------------------------------------------------
//
// Matrix multiplication with simple kernel (OpenCL vs serial CPU)
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "cl_wrapper2.h"

namespace chrono = std::chrono;

#if defined(SIMPLEST)
#define TS 1
#else
#define TS 16
#endif

#if defined(SIMPLEST) && !defined(MEASURE_NORMAL)
#define MEASURE_NORMAL 1
#endif

// naive version
const char *mmkernel = STRINGIFY(__kernel void simple_multiply(
    __global int *A, __global int *B, __global int *C, int AX, int AY, int BY) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int sum = 0;

  for (int k = 0; k < AY; k++)
    sum += A[row * AY + k] * B[k * BY + col];

  C[row * BY + col] = sum;
});

// simplest smoke test
#if defined(SIMPLEST)

#define AXVAL 3
#define AYVAL 2
#define BYVAL 1

// if we do not need to do ineffective normal multiplications,
// default sizes are big
#elif !defined(MEASURE_NORMAL)

#if !defined(AXVAL)
#define AXVAL 256 * 10
#endif

#if !defined(AYVAL)
#define AYVAL 256 * 8
#endif

#if !defined(BYVAL)
#define BYVAL 256 * 6
#endif

#else

#if !defined(AXVAL)
#define AXVAL 256 * 5
#endif

#if !defined(AYVAL)
#define AYVAL 256 * 4
#endif

#if !defined(BYVAL)
#define BYVAL 256 * 3
#endif

#endif

constexpr int BIG_AX = AXVAL;
constexpr int BIG_AY = AYVAL;
constexpr int BIG_BY = BYVAL;

// A[AX][AY] * B[AY][BY] = C[AX][BY]
void matrix_mult_normal(const int *A, const int *B, int *C, int AX, int AY,
                        int BY) {
  int i, j, k;
  for (i = 0; i < AX; i++) {
    for (j = 0; j < BY; j++) {
      int acc = 0;
      for (k = 0; k < AY; k++)
        acc += A[i * AY + k] * B[k * BY + j];
      C[i * BY + j] = acc;
    }
  }
}

// to exclude silly errors
void smoketest() {
  int a[3][2] = {{1, 2}, {3, 4}, {5, 6}};
  int b[2][1] = {{1}, {2}};
  int c[3][1];

  matrix_mult_normal(&a[0][0], &b[0][0], &c[0][0], 3, 2, 1);

  bool res = (c[0][0] == 5) && (c[1][0] == 11) && (c[2][0] == 17);
  if (!res) {
    std::cerr << "Smoke test failed!\n";
    std::cerr << "Wrong result:\n";
    for (int i = 0; i < 3; ++i)
      std::cerr << c[i][0] << " ";
    std::cerr << "\n";
    std::cerr << "Correct result:\n";
    std::cerr << 5 << " " << 11 << " " << 17 << "\n";
    abort();
  }
}

void matrix_rand_init(int *arr, int sz) {
  static std::mt19937_64 mt_source;
  std::uniform_int_distribution<int> dist(0, 10);
  for (int i = 0; i < sz; ++i)
    arr[i] = dist(mt_source);
}

int main() {
  smoketest();
  std::cout << "Welcome to matrix multiplication" << std::endl;
  std::cout << "[ " << BIG_AX << " x " << BIG_AY << " ] * [ " << BIG_AY << " x "
            << BIG_BY << " ]" << std::endl;

#if !defined(SIMPLEST)
  int(*a)[BIG_AY] = new int[BIG_AX][BIG_AY];
  int(*b)[BIG_BY] = new int[BIG_AY][BIG_BY];
  int(*c)[BIG_BY] = new int[BIG_AX][BIG_BY];
  int(*cref)[BIG_BY] = new int[BIG_AX][BIG_BY];

  matrix_rand_init(&a[0][0], BIG_AX * BIG_AY);
  matrix_rand_init(&b[0][0], BIG_AY * BIG_BY);

#else
  int a[BIG_AX][BIG_AY] = {{1, 2}, {3, 4}, {5, 6}};
  int b[BIG_AY][BIG_BY] = {{1}, {2}};
  int c[BIG_AX][BIG_BY];
  int cref[BIG_AX][BIG_BY];
#endif

  chrono::high_resolution_clock::time_point tstart, tfin;

#if MEASURE_NORMAL
  tstart = chrono::high_resolution_clock::now();
  matrix_mult_normal(&a[0][0], &b[0][0], &cref[0][0], BIG_AX, BIG_AY, BIG_BY);
  tfin = chrono::high_resolution_clock::now();
  std::cout
      << "Normal calculation time: "
      << chrono::duration_cast<chrono::milliseconds>(tfin - tstart).count()
      << std::endl;
#endif

  oclwrap2::ocl_app_t app;
  std::cout << "Selected platform: " << app.platform_version() << std::endl;
  std::cout << "Selected device: " << app.device_name() << std::endl;

  tstart = chrono::high_resolution_clock::now();

  int pidx = app.add_programm(mmkernel);
  int kidx = app.extract_kernel(pidx, "simple_multiply");

  size_t asz = BIG_AX * BIG_AY;
  size_t bsz = BIG_AY * BIG_BY;
  size_t csz = BIG_AX * BIG_BY;

  int abuf = app.add_buffer<int>(CL_MEM_READ_ONLY, &a[0][0], asz);
  int bbuf = app.add_buffer<int>(CL_MEM_READ_ONLY, &b[0][0], bsz);
  int cbuf = app.add_buffer<int>(CL_MEM_WRITE_ONLY, NULL, csz);

  app.set_kernel_buf_arg(kidx, 0, abuf);
  app.set_kernel_buf_arg(kidx, 1, bbuf);
  app.set_kernel_buf_arg(kidx, 2, cbuf);

  app.set_kernel_int_arg(kidx, 3, BIG_AX);
  app.set_kernel_int_arg(kidx, 4, BIG_AY);
  app.set_kernel_int_arg(kidx, 5, BIG_BY);

  size_t globalws[2] = {BIG_AX, BIG_BY};
  size_t localws[2] = {TS, TS};
  tfin = chrono::high_resolution_clock::now();
  std::cout
      << "Setup time: "
      << chrono::duration_cast<chrono::milliseconds>(tfin - tstart).count()
      << std::endl;

  // ocl multiplication
  tstart = chrono::high_resolution_clock::now();
  app.exec_kernel_nd(kidx, 2, globalws, localws);
  tfin = chrono::high_resolution_clock::now();
  std::cout
      << "OCL calculation time: "
      << chrono::duration_cast<chrono::milliseconds>(tfin - tstart).count()
      << std::endl;

  tstart = chrono::high_resolution_clock::now();
  app.read_buffer<int>(cbuf, &c[0][0], csz);
  tfin = chrono::high_resolution_clock::now();
  std::cout
      << "Teardown time: "
      << chrono::duration_cast<chrono::milliseconds>(tfin - tstart).count()
      << std::endl;

#if defined(SIMPLEST)
  std::cout << "COCL: {" << c[0][0] << ", " << c[1][0] << ", " << c[2][0]
            << "} " << std::endl;
  std::cout << "CREF: {" << cref[0][0] << ", " << cref[1][0] << ", "
            << cref[2][0] << "} " << std::endl;
#endif

#if MEASURE_NORMAL
  for (int i = 0; i < BIG_AX; ++i)
    for (int j = 0; j < BIG_BY; ++j)
      if (cref[i][j] != c[i][j]) {
        std::cerr << i << ": " << c[i][j] << " != " << cref[i][j] << std::endl;
        abort();
      }
  std::cout << "Check against normal passed, matrices multiplied ok"
            << std::endl;
#endif

#if !defined(SIMPLEST)
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] cref;
#endif
}
