//------------------------------------------------------------------------------
//
// Vector addition, SYCL way, with malloc_device
// Explicit wait on queue is basic synchronization mechanism
// Try -DNOWAIT to trigger strange bugs
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <CL/sycl.hpp>

#include <iostream>
#include <vector>

constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;

constexpr int LIST_SIZE = 1024 * 1024 * 2;
using arr_t = std::vector<cl::sycl::cl_int>;

// class is used for kernel name
template <typename T> class simple_vector_add;

void print_info(std::ostream &os, const cl::sycl::queue &deviceQueue) {
  auto device = deviceQueue.get_device();
  os << device.get_info<cl::sycl::info::device::name>() << "\n";
  os << "Driver version: "
     << device.get_info<cl::sycl::info::device::driver_version>() << "\n";
  os << device.get_info<cl::sycl::info::device::opencl_c_version>() << "\n";
}

template <typename T>
void process_buffers(T const *pa, T const *pb, T *pc, size_t sz,
                     cl::sycl::queue &deviceQueue);

int main() {
  arr_t A(LIST_SIZE), B(LIST_SIZE), C(LIST_SIZE);
  std::cout << "Welcome to vector addition" << std::endl;

  try {
    cl::sycl::gpu_selector GPsel;
    cl::sycl::queue Q{GPsel};
    print_info(std::cout, Q);

    std::cout << "Initializing" << std::endl;
    for (int i = 0; i < LIST_SIZE; i++) {
      A[i] = i;
      B[i] = LIST_SIZE - i;
    }

    std::cout << "Calculating" << std::endl;
    process_buffers(A.data(), B.data(), C.data(), LIST_SIZE, Q);
  } catch (cl::sycl::exception const &err) {
    std::cerr << "ERROR: " << err.what() << ":\n";
    return -1;
  }
  std::cout << "Everything is correct" << std::endl;
}

template <typename T>
void process_buffers(T const *pa, T const *pb, T *pc, size_t sz,
                     cl::sycl::queue &deviceQueue) {
  int *A = cl::sycl::malloc_device<T>(sz, deviceQueue);
  int *B = cl::sycl::malloc_device<T>(sz, deviceQueue);
  int *C = cl::sycl::malloc_device<T>(sz, deviceQueue);

  // kernels to copy to device
  deviceQueue.submit(
      [&](cl::sycl::handler &cgh) { cgh.memcpy(A, pa, sz * sizeof(T)); });
  deviceQueue.submit(
      [&](cl::sycl::handler &cgh) { cgh.memcpy(B, pb, sz * sizeof(T)); });
#ifndef NOWAIT
  deviceQueue.wait();
#endif

  // vector addition
  cl::sycl::range<1> numOfItems{sz};
  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto kern = [A, B, C](cl::sycl::id<1> wiID) {
      C[wiID] = A[wiID] + B[wiID];
    };
    cgh.parallel_for<class simple_vector_add<T>>(numOfItems, kern);
  });
#ifndef NOWAIT
  deviceQueue.wait();
#endif

  // copy back
  deviceQueue.submit(
      [&](cl::sycl::handler &cgh) { cgh.memcpy(pc, C, sz * sizeof(T)); });
#ifndef NOWAIT
  deviceQueue.wait();
#endif

  std::cout << "Checking with host results" << std::endl;

  // trying to access device memory on host
#ifdef TRYACC
  std::cout << C[0] << std::endl;
#endif

  for (int i = 0; i < LIST_SIZE; ++i)
    if (pc[i] != pa[i] + pb[i]) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << pc[i] << " != " << pa[i] + pb[i] << "\n";
      abort();
    }
}
