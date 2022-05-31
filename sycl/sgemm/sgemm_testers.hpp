//------------------------------------------------------------------------------
//
// Generic code to test different variants of matrix mult
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//  -DMEASURE_NORMAL : measure with normal host code
//  -DMULT_INEFF : in host code, not transpose matrix first for cache effects
//
// Options to control things:
// -ax=<n>, -ay=<m>, -by=<k> : matrix sizes
// -lsz=<l> : amount of local address space
// -vis : visualize matrices (use wisely) available only in measure_normal
// -quiet : quiet mode (say for gnuplot stuff), output only GPU time or errors
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

constexpr int DEF_BLOCK = 256;
constexpr int DEF_AX = 5;
constexpr int DEF_AY = 4;
constexpr int DEF_BY = 3;
constexpr int DEF_LSZ = 8;

namespace sycltesters {

namespace sgemm {
struct Config {
  size_t Ax, Ay, By, Block;
  unsigned Lsz;
  bool Vis = false, Quiet = false;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;

  options::Parser OptParser;
  OptParser.template add<int>(
      "ax", DEF_AX, "size X of matrix A in A * B in bsz-element blocks");
  OptParser.template add<int>(
      "ay", DEF_AY, "size Y of matrix A in A * B in bsz-element blocks");
  OptParser.template add<int>(
      "by", DEF_BY, "size Y of matrix B in A * B in bsz-element blocks");
  OptParser.template add<int>("lsz", DEF_LSZ, "local size");
  OptParser.template add<int>("bsz", DEF_BLOCK,
                              "size of block (matrix size multiple)");
  OptParser.template add<int>("vis", 0, "visualize matrices");
  OptParser.template add<int>("quiet", 0, "quiet mode for bulk runs");
  OptParser.parse(argc, argv);

  Cfg.Block = OptParser.template get<int>("bsz");
  Cfg.Ax = OptParser.template get<int>("ax") * Cfg.Block;
  Cfg.Ay = OptParser.template get<int>("ay") * Cfg.Block;
  Cfg.By = OptParser.template get<int>("by") * Cfg.Block;
  Cfg.Lsz = OptParser.template get<int>("lsz");
  Cfg.Vis = OptParser.exists("vis");

  if (OptParser.exists("quiet")) {
    Cfg.Quiet = true;
    Cfg.Vis = false;
    qout.set(Cfg.Quiet);
  }

  return Cfg;
}

inline void dump_config_info(Config &Cfg) {
  if (Cfg.Vis)
    qout << "Visual mode" << std::endl;
  qout << "Using sizes: " << Cfg.Ax << ", " << Cfg.Ay << ", " << Cfg.By
       << std::endl;
  qout << "Block size: " << Cfg.Block << std::endl;
  qout << "Local size: " << Cfg.Lsz << std::endl;
}

} // namespace sgemm

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
#if !defined(MULT_INEFF)
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
  try {
    auto Cfg = sgemm::read_config(argc, argv);
    qout << "Welcome to matrix multiplication" << std::endl;
    sgemm::dump_config_info(Cfg);

    auto Q = set_queue();
    print_info(qout, Q.get_device());

    qout << "Initializing" << std::endl;
    using Ty = typename MMChildT::type;
    std::vector<Ty> A(Cfg.Ax * Cfg.Ay), B(Cfg.Ay * Cfg.By);
    rand_initialize(A.data(), A.size(), MINF, MAXF);
    rand_initialize(B.data(), B.size(), MINF, MAXF);

#ifdef MEASURE_NORMAL
    qout << "Calculating host" << std::endl;
    MatrixMultHost<Ty> MMultH{Q}; // Q unused for this derived class
    MatrixMultTester<Ty> TesterH{MMultH, A.data(), B.data(),
                                 Cfg.Ax, Cfg.Ay,   Cfg.By};
    auto ElapsedH = TesterH.calculate();
    qout << "Measured host time: " << ElapsedH.first / msec_per_sec << "\n";
#endif

    MMChildT MMult{Q, Cfg};

    MatrixMultTester<Ty> Tester{MMult,  A.data(), B.data(),
                                Cfg.Ax, Cfg.Ay,   Cfg.By};

    qout << "Calculating gpu" << std::endl;
    auto Elapsed = Tester.calculate();
    auto ExecTime = Elapsed.second / nsec_per_sec;

    qout << "Measured time: " << Elapsed.first / msec_per_sec << std::endl;
    qout << "Pure execution time: " << ExecTime << std::endl;

    // only things that shall occur on console in quiet mode: Ax and time
    // we may run this in the loop
    if (Cfg.Quiet) {
      qout.set(!Cfg.Quiet);
      qout << Cfg.Ax << " " << ExecTime << std::endl;
      qout.set(Cfg.Quiet);
    }

#if defined(MEASURE_NORMAL) || defined(VERIFY)
    Ty *HostData = TesterH.getref();
    Ty *GPUData = Tester.getref();

    if (Cfg.Vis) {
      dump_matrix(qout, "A", Tester.getA(), Cfg.Ax, Cfg.Ay);
      dump_matrix(qout, "B", Tester.getB(), Cfg.Ay, Cfg.By);
      dump_matrix(qout, "Host result", HostData, Cfg.Ax, Cfg.By);
      dump_matrix(qout, "GPU result", GPUData, Cfg.Ax, Cfg.By);
    }

#if defined(VERIFY)
    // verification with host result
    for (int I = 0; I < Cfg.Ax * Cfg.By; ++I)
      if (HostData[I] != GPUData[I]) {
        std::cerr << "Mismatch at: " << I << std::endl;
        std::cerr << HostData[I] << " vs " << GPUData[I] << std::endl;
        std::terminate();
      }
#endif // VERIFY
#endif // MEASURE_NORMAL || VERIFY
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
  qout << "Everything is correct" << std::endl;
}

} // namespace sycltesters
