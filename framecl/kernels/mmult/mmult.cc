#include <iostream>
#include <string>

#include "framecl.hpp"

// clang-format off
#define cimg_use_jpeg
#include "CImg.h"
// clang-format on

constexpr int DEFAX = 256 * 5;
constexpr int DEFAY = 256 * 4;
constexpr int DEFBY = 256 * 3;
constexpr int DEFLOCALSZ = 16;

// slow reference host mult: A[AX][AY] * B[AY][BY] = C[AX][BY]
template <typename T>
void matrix_mult_ref(T &A, T &B, T &C, int AX, int AY, int BY) {
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

int mmultmain(int argc, char **argv) {
  framecl::optparser_t opts;

  opts.add<int>("ax", DEFAX, "size of first matrix, X part");
  opts.add<int>("ay", DEFAY, "size of first matrix, Y part");
  opts.add<int>("by", DEFBY, "size of second matrix, Y part");
  opts.add<int>("lsz", DEFLOCALSZ, "local work space size");

  opts.parse(argc, argv);

  opts.require_platform("This program requires platform specification. Use "
                        "--list for available platforms");
  opts.require_program("This program needs external cl program file. It shall "
                       "contain 'matrix_multiply' kernel");

  int ax = opts.check<int>(
      "ax", [](int ax) { return ax > 0; }, "all matrix sizes shall be > 0");
  int ay = opts.check<int>(
      "ay", [](int ay) { return ay > 0; }, "all matrix sizes shall be > 0");
  int by = opts.check<int>(
      "by", [](int by) { return by > 0; }, "all matrix sizes shall be > 0");
  int lsz = opts.check<int>(
      "lsz", [](int lsz) { return lsz > 0; }, "local ws size shall be > 0");

  if (!opts.quiet()) {
    std::cout << "Welcome to matrix multiplication" << std::endl;
    std::cout << "[ " << ax << " x " << ay << " ] * [ " << ay << " x " << by
              << " ]" << std::endl;
  }

  int asz = ax * ay;
  int bsz = ay * by;
  int csz = ax * by;

  cl::size_type globalx = ax;
  cl::size_type globaly = by;
  cl::size_type localsz = lsz;

  framecl::context_t ctx(opts);
  framecl::program_t prog(ctx, opts, [lsz](std::string &s) {
    std::ostringstream os;
    os << "#define TS " << lsz << "\n";
    os << s;
    s = os.str();
  });

  cl::NDRange offset{0, 0}, global{globalx, globaly}, local{localsz, localsz};

  framecl::run_params_t parms{offset, global, local};
  framecl::functor_t<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int>
      matrix_multiply(prog, parms, "matrix_multiply");

  framecl::buffer_t<cl_int> bufA{ctx, asz}, bufB{ctx, bsz}, bufC{ctx, csz},
      bufCRef{ctx, csz};

  rand_init(bufA, asz, 0, 10);
  rand_init(bufB, bsz, 0, 10);

  framecl::task_t writeA(framecl::task::write, bufA);
  framecl::task_t writeB(framecl::task::write, bufB);
  framecl::task_t execF(framecl::task::process, matrix_multiply, bufA.base(),
                        bufB.base(), bufC.base(), ax, ay, by);
  framecl::task_t readC(framecl::task::read, bufC);

  // clang-format off
  framecl::depgraph_t dg(ctx, {
    {&writeA},
    {&writeB},
    {&execF, &writeA, &writeB},
    {&readC, &execF}
  });
  // clang-format on

  if (opts.verbose()) {
    std::cout << "Dep graph for tasks:" << std::endl;
    dg.dump(std::cout);
    std::cout << std::endl;
  }

  dg.execute();

  if (opts.check()) {
    std::cout << "Cross-check with non-ocl results... ";
    matrix_mult_ref(bufA, bufB, bufCRef, ax, ay, by);

    for (int i = 0; i < csz; ++i)
      if (bufC[i] != bufCRef[i]) {
        std::cout << "failed at i = " << i << ", C[i] = " << bufC[i]
                  << ", CREF[i] = " << bufCRef[i] << std::endl;
        throw std::logic_error("Check failed");
      }
    std::cout << "ok" << std::endl;
  }

  return 0;
}

int main(int argc, char **argv) {
  try {
    return mmultmain(argc, argv);
  } catch (cl::Error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    process_error_t pe(e.err());
    pe(std::cerr);
    std::cerr << std::endl;
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
  return -1;
}