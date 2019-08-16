//-----------------------------------------------------------------------------
//
// Adding vectors in OpenCL, C++ way
//
//-----------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <iostream>
#include <stdexcept>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#define STRINGIFY(...) #__VA_ARGS__

constexpr size_t LIST_SIZE = 1024;

using arr_t = std::vector<long>;

// ---------------------------------- OpenCL ---------------------------------
const char *vakernel = STRINGIFY(
__kernel void vector_add(
    __global int *A, __global int *B, __global int *C) {
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
});
// ---------------------------------- OpenCL ---------------------------------

class ocl_ctx_t {
  cl::Context context;
  cl::CommandQueue queue;

public:
  ocl_ctx_t(cl_mem_flags flags) : context(flags), queue(context) {}

  template <typename T>
  void process_buffers(T const *pa, T const *pb, T *pc, size_t sz);
};

int main() {
  arr_t A(LIST_SIZE), B(LIST_SIZE), C(LIST_SIZE);

  try {
    ocl_ctx_t ct(CL_DEVICE_TYPE_GPU);

    for (int i = 0; i < LIST_SIZE; i++) {
      A[i] = i;
      B[i] = LIST_SIZE - i;
    }

    ct.process_buffers(A.data(), B.data(), C.data(), LIST_SIZE);

    for (int i = 0; i < LIST_SIZE; ++i)
      if (C[i] != A[i] + B[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << C[i] << " != " << A[i] + B[i] << "\n";
        abort();
      }

    std::cout << "Everything is correct" << std::endl;
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << ":\n";
    return -1;
  }
}

template <typename T>
void ocl_ctx_t::process_buffers(T const *pa, T const *pb, T *pc, size_t sz) {
  size_t bufsz = sz * sizeof(T);

  cl::Buffer abuf(context, CL_MEM_READ_ONLY, bufsz);
  cl::Buffer bbuf(context, CL_MEM_READ_ONLY, bufsz);
  cl::Buffer cbuf(context, CL_MEM_WRITE_ONLY, bufsz);

  cl::copy(queue, pa, pa + sz, abuf);
  cl::copy(queue, pb, pb + sz, bbuf);

  cl::Program program(std::string(vakernel), true);
  cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> add_vecs(program,
                                                               "vector_add");
  cl::NDRange global(sz);
  cl::NDRange local(64);

  add_vecs(cl::EnqueueArgs(queue, global, local), abuf, bbuf, cbuf);

  cl::copy(queue, cbuf, pc, pc + sz);
}
