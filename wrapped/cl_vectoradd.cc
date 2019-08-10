#include <iostream>

#include "cl_wrapper2.h"

constexpr int LIST_SIZE = 1024;

const char *vakernel = STRINGIFY(__kernel void vector_add(
    __global int *A, __global int *B, __global int *C) {
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
});

void sum_vectors_normal(const int *A, const int *B, int *C, int sz) {
  int i;

  for (i = 0; i < sz; i++)
    C[i] = A[i] + B[i];
}

void sum_vectors_ocl(oclwrap2::ocl_app_t &app, int kidx, const int *A,
                     const int *B, int *C, int sz) {
  int abuf = app.add_buffer<int>(CL_MEM_READ_ONLY, A, sz);
  int bbuf = app.add_buffer<int>(CL_MEM_READ_ONLY, B, sz);
  int cbuf = app.add_buffer<int>(CL_MEM_WRITE_ONLY, NULL, sz);

  app.set_kernel_buf_arg(kidx, 0, abuf);
  app.set_kernel_buf_arg(kidx, 1, bbuf);
  app.set_kernel_buf_arg(kidx, 2, cbuf);
  size_t globalsize = sz;
  size_t localsize = 256;
  app.exec_kernel_nd(kidx, 1, &globalsize, &localsize);
  app.read_buffer<int>(cbuf, C, sz);
}

int main(void) {
  // Create two input vectors
  std::vector<int> A(LIST_SIZE), B(LIST_SIZE), C(LIST_SIZE), CREF(LIST_SIZE);

  for (int i = 0; i < LIST_SIZE; i++) {
    A[i] = i;
    B[i] = LIST_SIZE - i;
  }

  sum_vectors_normal(&A[0], &B[0], &CREF[0], LIST_SIZE);

  oclwrap2::ocl_app_t app;
  std::cout << "Selected platform: " << app.platform_version() << std::endl;
  std::cout << "Selected device: " << app.device_name() << std::endl;

  int pidx = app.add_programm(vakernel);
  int kidx = app.extract_kernel(pidx, "vector_add");

  sum_vectors_ocl(app, kidx, &A[0], &B[0], &C[0], LIST_SIZE);

#ifdef DISPLAY
  for (int i = 0; i < LIST_SIZE; i++)
    printf("%d + %d = %d\n", A[i], B[i], C[i]);
#endif

  for (int i = 0; i < LIST_SIZE; i++)
    if (C[i] != CREF[i]) {
      std::cerr << i << ": " << C[i] << " != " << CREF[i] << std::endl;
      abort();
    }

  return 0;
}
