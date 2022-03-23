//------------------------------------------------------------------------------
//
// Generic code to test different variants of matrix mult
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//  -DMEASURE_NORMAL : measure with normal host code
//  -DMULT_TRANSPOSE : in host code, transpose matrix first for cache effects
//
// Options to control things:
// -ax=<n>, -ay=<m>, -by=<k> : matrix sizes
// -lsz=<l> : amount of local address space
// -vis=1 : visualize matrices (use wisely) available only in measure_normal
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <bit>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

// problems with boost in OneAPI console on Windows
#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#include "testers.hpp"

constexpr int DEF_AX = 256 * 5;
constexpr int DEF_AY = 256 * 4;
constexpr int DEF_BY = 256 * 3;
constexpr int DEF_LSZ = 1;

namespace sycltesters {

template <typename T> class MatrixMult {
  cl::sycl::queue DeviceQueue_;

public:
  using type = T;
  MatrixMult(cl::sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(const T *A, const T *B, T *C, size_t AX,
                              size_t AY, size_t BY) = 0;
  cl::sycl::queue &Queue() { return DeviceQueue_; }
  const cl::sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~MatrixMult() {}
};

template <typename T> struct MatrixMultHost : public MatrixMult<T> {
  void mmult_normal(const T *A, const T *B, T *C, size_t AX, size_t AY,
                    size_t BY) {
    int i, j, k;
    for (i = 0; i < AX; i++) {
      for (j = 0; j < BY; j++) {
        T acc = 0;
        for (k = 0; k < AY; k++)
          acc += A[i * AY + k] * B[k * BY + j];
        C[i * BY + j] = acc;
      }
    }
  }
  void mmult_transpose(const T *A, const T *B, T *C, size_t AX, size_t AY,
                       size_t BY) {
    std::vector<T> tmp(BY * AY);
    for (int i = 0; i < AY; i++)
      for (int j = 0; j < BY; j++)
        tmp[j * AY + i] = B[i * BY + j];

    for (int i = 0; i < AX; i++)
      for (int j = 0; j < BY; j++) {
        T acc = 0;
        for (int k = 0; k < AY; k++)
          acc += A[i * AY + k] * tmp[j * AY + k];
        C[i * BY + j] = acc;
      }
  }

public:
  MatrixMultHost(cl::sycl::queue &DeviceQueue) : MatrixMult<T>(DeviceQueue) {}
  EvtRet_t operator()(const T *A, const T *B, T *C, size_t AX, size_t AY,
                      size_t BY) override {
#ifdef MULT_TRANSPOSE
    mmult_transpose(A, B, C, AX, AY, BY);
#else
    mmult_normal(A, B, C, AX, AY, BY);
#endif
    return {}; // nothing to construct as event
  }
};

// we want only floats -10 .. 10
constexpr int MINF = -10;
constexpr int MAXF = 10;

template <typename T> class MatrixMultTester {
  MatrixMult<T> &Multiply_;
  Timer Timer_;
  size_t AX_, AY_, BY_;
  const T *A_;
  const T *B_;
  std::vector<T> C_;

public:
  MatrixMultTester(MatrixMult<T> &Multiply, const T *A, const T *B, size_t AX,
                   size_t AY, size_t BY)
      : Multiply_(Multiply), AX_(AX), AY_(AY), BY_(BY), A_(A), B_(B),
        C_(AX * BY) {}

  std::pair<unsigned, unsigned> calculate() {
    unsigned EvtTiming = 0;
    Timer_.start();
    EvtRet_t Ret = Multiply_(A_, B_, C_.data(), AX_, AY_, BY_);
    EvtTiming += getTime(Ret);
    Timer_.stop();
    return {Timer_.elapsed(), EvtTiming};
  }

  const T *getA() const { return A_; }
  const T *getB() const { return B_; }
  T *getref() { return C_.data(); }
};

template <typename T>
void dump_matrix(std::ostream &Os, std::string Name, T *M, int X, int Y) {
  Os << Name << ":\n";
  for (int I = 0; I < X; ++I) {
    for (int J = 0; J < Y; ++J)
      Os << M[I * Y + J] << " ";
    Os << "\n";
  }
}

template <typename T>
void rand_initialize(T *Arr, size_t Sz, int min, int max) {
  Dice DZERO(0, 100);
  Dice D(min, max);
  // most zeroes for floating point to reduce probability of overflow
  for (int I = 0; I < Sz; ++I)
    Arr[I] = (DZERO() < 50) ? D() : 0;
}

template <typename MMChildT> void test_sequence(int argc, char **argv) {
  std::cout << "Welcome to matrix multiplication" << std::endl;

  try {
    size_t Ax, Ay, By, Lsz;

    optparser_t OptParser;
    OptParser.template add<int>("ax", DEF_AX);
    OptParser.template add<int>("ay", DEF_AY);
    OptParser.template add<int>("by", DEF_BY);
    OptParser.template add<int>("lsz", DEF_LSZ);
#ifdef MEASURE_NORMAL
    OptParser.template add<int>("vis", 0);
#endif
    OptParser.parse(argc, argv);

    Ax = OptParser.template get<int>("ax");
    Ay = OptParser.template get<int>("ay");
    By = OptParser.template get<int>("by");
    Lsz = OptParser.template get<int>("lsz");
#ifdef MEASURE_NORMAL
    int Vis = OptParser.template get<int>("vis");
#endif

    std::cout << "Using sizes: " << Ax << ", " << Ay << ", " << By << std::endl;
    std::cout << "Local size: " << Lsz << std::endl;
#ifdef MEASURE_NORMAL
    if (Vis)
      std::cout << "Visual mode" << std::endl;
#endif

    auto Q = set_queue();
    print_info(std::cout, Q.get_device());

    std::cout << "Initializing" << std::endl;
    using Ty = typename MMChildT::type;
    std::vector<Ty> A(Ax * Ay), B(Ay * By);
    rand_initialize(A.data(), A.size(), MINF, MAXF);
    rand_initialize(B.data(), B.size(), MINF, MAXF);

#ifdef MEASURE_NORMAL
    std::cout << "Calculating host" << std::endl;
    MatrixMultHost<Ty> MMultH{Q}; // Q unused for this derived class
    MatrixMultTester<Ty> TesterH{MMultH, A.data(), B.data(), Ax, Ay, By};
    auto ElapsedH = TesterH.calculate();
    std::cout << "Measured host time: " << ElapsedH.first << std::endl;
#endif

    MMChildT MMult{Q, Lsz};

    MatrixMultTester<Ty> Tester{MMult, A.data(), B.data(), Ax, Ay, By};

    std::cout << "Calculating gpu" << std::endl;
    auto Elapsed = Tester.calculate();

    std::cout << "Measured time: " << Elapsed.first / 1000.0 << std::endl;
    std::cout << "Pure execution time: " << Elapsed.second / 1000000000.0
              << std::endl;

#ifdef MEASURE_NORMAL
    // verification with host result
    Ty *HostData = TesterH.getref();
    Ty *GPUData = Tester.getref();

    if (Vis) {
      dump_matrix(std::cout, "A", Tester.getA(), Ax, Ay);
      dump_matrix(std::cout, "B", Tester.getB(), Ay, By);
      dump_matrix(std::cout, "Host result", HostData, Ax, By);
      dump_matrix(std::cout, "GPU result", GPUData, Ax, By);
    }

    for (int I = 0; I < Ax * By; ++I)
      if (HostData[I] != GPUData[I]) {
        std::cout << "Mismatch at: " << I << std::endl;
        std::cout << HostData[I] << " vs " << GPUData[I] << std::endl;
        return;
      }
#endif
  } catch (cl::sycl::exception const &err) {
    std::cerr << "SYCL ERROR: " << err.what() << "\n";
    abort();
  } catch (std::exception const &err) {
    std::cerr << "Exception: " << err.what() << "\n";
    abort();
  } catch (...) {
    std::cerr << "Unknown error\n";
    abort();
  }
  std::cout << "Everything is correct" << std::endl;
}

} // namespace sycltesters
