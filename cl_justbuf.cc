#include <iostream>
#include <stdexcept>	
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

class ocl_ctx_t {
  cl::Context context;
  cl::CommandQueue queue;
public:
  ocl_ctx_t(cl_mem_flags flags) : context(flags), queue(context) {}
  void process_buffers();
};

int main() {
  try {
    ocl_ctx_t ct(CL_DEVICE_TYPE_GPU);
    ct.process_buffers();
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << ":\n";    
    return -1;
  }
}

constexpr size_t BUFSZ = 128;

class buf_write_error : public std::runtime_error {
  const char *msg_;
  int Bi_, i_;
public:
  buf_write_error(const char *msg, int Bi, int i) : std::runtime_error(msg), Bi_(Bi), i_(i) {}
  const char* what() const noexcept override {
    return std::runtime_error::what();
  }
};

void ocl_ctx_t::process_buffers() {
  std::vector<int> A(BUFSZ), B(BUFSZ);
  for (int i = 0; i < BUFSZ; ++i)
    A[i] = i * i;

  cl::Buffer oclbuf(context, CL_MEM_READ_WRITE, BUFSZ * sizeof(int));

  // A --> oclbuf
  cl::copy(queue, A.begin(), A.end(), oclbuf);

  // oclbuf --> B
  cl::copy(queue, oclbuf, B.begin(), B.end());

  for (int i = 0; i < BUFSZ; ++i)
    if (B[i] != i * i)
      throw buf_write_error("Buf checking error", B[i], i);

  std::cout << "Everything is correct" << std::endl;
}

