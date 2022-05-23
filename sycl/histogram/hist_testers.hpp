//------------------------------------------------------------------------------
//
// Generic code to test different variants of histogram
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//  -DMEASURE_NORMAL : measure with normal host code
//
// Options to control things:
// -bsz=<bsz> : block size
// -sz=<sz> : data size (in bsz-units)
// -hsz=<hsz> : number of buckets
// -gsz=<g> : global iteration space (in bsz-units)
// -lsz=<l> : local iteration space
// -vis=1 : visualize hist (use wisely) available only in measure_normal
// -zero=1 : fill hist with zeroes for debug
//
// Special visualization part:
// -img=path : path to image to build realistics hist
// -bwidth=<bwidth> : width of columns in visualized hist
//
// Try:
// > hist_naive.exe -sz=16 -bsz=1 -gsz=8 -hsz=4 -lsz=2 -vis=1 -zero=1
// > hist_naive.exe -img=..\favn.jpg
//
// Perf measures for TGLLP:
// > hist_local.exe -sz=200000 -bsz=1000 -zero=1
// pure exec time on GPU ~= 0.03
// > hist_naive.exe -sz=200000 -bsz=1000 -zero=1
// pure exec time on GPU ~= 0.3
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>
#include <vector>

#include <CL/sycl.hpp>

#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#define CIMG_ENABLE

#ifdef CIMG_ENABLE
#include "drawer.hpp"
#endif
#include "testers.hpp"

// we will need global and local atomic references
template <typename T>
using local_atomic_ref = cl::sycl::ext::oneapi::atomic_ref<
    T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::work_group,
    cl::sycl::access::address_space::local_space>;

template <typename T>
using global_atomic_ref = cl::sycl::ext::oneapi::atomic_ref<
    T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::system,
    cl::sycl::access::address_space::global_space>;

namespace sycltesters {

constexpr int DEF_BSZ = 128;
constexpr int DEF_SZ = 1024; // data size in blocks
constexpr int DEF_HSZ = 260;
constexpr int DEF_GSZ = 64; // global iteration space in blocks
constexpr int DEF_LSZ = 32;
constexpr int DEF_BWIDTH = 2;
constexpr int DEF_VIS = 0;
constexpr int DEF_ZEROOUT = 0;
constexpr int DEF_DETAILED = 0;

namespace hist {

struct Config {
  int Block, Sz, HistSz, GlobSz, LocSz, Vis, BWidth, Zero, Detailed;
  std::string Image;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  optparser_t OptParser;
  OptParser.template add<int>("bsz", DEF_BSZ, "block size");
  OptParser.template add<int>("sz", DEF_SZ,
                              "data size (in bsz-element blocks)");
  OptParser.template add<int>("hsz", DEF_HSZ, "number of buckets");
  OptParser.template add<int>("gsz", DEF_GSZ,
                              "global iteration space (in bsz-element blocks)");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
  OptParser.template add<int>("vis", 0, "pass 1 to visualize matrices");
  OptParser.template add<std::string>("img", "",
                                      "pass image path to load image");
  OptParser.template add<int>("bwidth", DEF_BWIDTH,
                              "bin width for hist visualization");
  OptParser.template add<int>("zero", DEF_ZEROOUT, "fill data with zeroes");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.parse(argc, argv);

  Cfg.Image = OptParser.template get<std::string>("img");
  Cfg.Block = OptParser.template get<int>("bsz");
  Cfg.Sz = OptParser.template get<int>("sz") * Cfg.Block;
  Cfg.HistSz = OptParser.template get<int>("hsz");
  Cfg.GlobSz = OptParser.template get<int>("gsz") * Cfg.Block;
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.Vis = OptParser.template get<int>("vis");
  Cfg.BWidth = OptParser.template get<int>("bwidth");
  Cfg.Zero = OptParser.template get<int>("zero");
  Cfg.Detailed = OptParser.template get<int>("detailed");

  if (Cfg.Block < 1 || Cfg.Sz < 1 || Cfg.HistSz < 1 || Cfg.GlobSz < 1 ||
      Cfg.LocSz < 1) {
    std::cerr << "Wrong parameters (expect all sizes >= 1)\n";
    std::terminate();
  }

#ifndef CIMG_ENABLE
  if (!Cfg.Image.empty()) {
    std::cerr << "You need build with CImg support to use this option\n";
    std::terminate();
  }
#endif

  std::cout << "Block size: " << Cfg.Block << std::endl;
  std::cout << "Data size: " << Cfg.Sz << std::endl;
  std::cout << "Histogram size: " << Cfg.HistSz << std::endl;
  std::cout << "Global size: " << Cfg.GlobSz << std::endl;
  std::cout << "Local size: " << Cfg.LocSz << std::endl;
  if (Cfg.Vis)
    std::cout << "Visual mode" << std::endl;
  if (Cfg.Detailed)
    std::cout << "Detailed events" << std::endl;
  return Cfg;
}
} // namespace hist

template <typename T> class Histogramm {
  cl::sycl::queue DeviceQueue_;

public:
  using type = T;
  Histogramm(cl::sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(const T *Data, T *Bins, size_t NumData,
                              size_t NumBins, EBundleTy ExeBundle) = 0;
  cl::sycl::queue &Queue() { return DeviceQueue_; }
  const cl::sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~Histogramm() {}
};

template <typename T> struct HistogrammHost : public Histogramm<T> {
  HistogrammHost(cl::sycl::queue &DeviceQueue) : Histogramm<T>(DeviceQueue) {}
  EvtRet_t operator()(const T *Data, T *Bins, size_t NumData, size_t NumBins,
                      EBundleTy ExeBundle) override {
    for (int i = 0; i < NumData; i++) {
      assert(Data[i] < NumBins);
      Bins[Data[i]] += 1;
    }
    return {}; // nothing to construct as event
  }
};

template <typename T> class HistogrammTester {
  Histogramm<T> &Hist_;
  Timer Timer_;
  const T *Data_;
  size_t NumData_, NumBins_;
  std::vector<T> Bins_;
  EBundleTy ExeBundle_;

public:
  HistogrammTester(Histogramm<T> &Hist, const T *Data, size_t NumData,
                   size_t NumBins, EBundleTy ExeBundle)
      : Hist_(Hist), Data_(Data), NumData_(NumData), NumBins_(NumBins),
        Bins_(NumBins), ExeBundle_(ExeBundle) {}

  std::pair<unsigned, unsigned> calculate(hist::Config Cfg) {
    Timer_.start();
    EvtRet_t Ret = Hist_(Data_, Bins_.data(), NumData_, NumBins_, ExeBundle_);
    Timer_.stop();
    auto EvtTiming = getTime(Ret, Cfg.Detailed ? false : true);
    return {Timer_.elapsed(), EvtTiming};
  }

  const T *data() const { return Data_; }
  auto dataBins() { return Bins_.data(); }
  auto sizeBins() { return Bins_.size(); }
  auto beginBins() { return Bins_.begin(); }
  auto endBins() { return Bins_.end(); }
};

template <typename T>
void dump_hist(std::ostream &Os, std::string Name, const T *Data, int Sz) {
  Os << Name << ":\n";
  std::ostream_iterator<T> OsIt(Os, " ");
  std::copy(Data, Data + Sz, OsIt);
  Os << "\n";
}

template <typename HistChildT, typename Ty>
HistogrammTester<Ty> single_hist_sequence(cl::sycl::queue &Q, hist::Config Cfg,
                                          Ty *Data, EBundleTy ExeBundle) {
#if defined(MEASURE_NORMAL)
  std::cout << "Calculating host" << std::endl;
  HistogrammHost<Ty> HistH{Q}; // Q unused for this derived class
  HistogrammTester<Ty> TesterH{HistH, Data, Cfg.Sz, Cfg.HistSz,
                               /* unused */ ExeBundle};
  auto ElapsedH = TesterH.calculate(Cfg);
  std::cout << "Measured host time: " << ElapsedH.first / msec_per_sec
            << std::endl;
  Ty *HostData = TesterH.dataBins();
  if (Cfg.Vis)
    dump_hist(std::cout, "Host result", HostData, Cfg.HistSz);
#endif

  HistChildT Hist{Q, Cfg.GlobSz, Cfg.LocSz};

  HistogrammTester<Ty> Tester{Hist, Data, Cfg.Sz, Cfg.HistSz, ExeBundle};

  std::cout << "Calculating gpu" << std::endl;
  auto Elapsed = Tester.calculate(Cfg);

  std::cout << "Measured time: " << Elapsed.first / msec_per_sec << std::endl
            << "Pure execution time: " << Elapsed.second / nsec_per_sec
            << std::endl;

  Ty *GPUData = Tester.dataBins();

  if (Cfg.Vis) {
    dump_hist(std::cout, "Data: ", Tester.data(), Cfg.Sz);
    dump_hist(std::cout, "GPU result", GPUData, Cfg.HistSz);
  }

#if defined(MEASURE_NORMAL) && defined(VERIFY)
  // verification with host result
  auto MisPoint = std::mismatch(HostData, HostData + Cfg.HistSz, GPUData);
  if (MisPoint.first != HostData + Cfg.HistSz) {
    ptrdiff_t I = MisPoint.first - HostData;
    std::cout << "Mismatch at: " << I << std::endl;
    std::cout << *MisPoint.first << " vs " << *MisPoint.second << std::endl;
    std::terminate();
  }
#endif
  return Tester;
}

// implemented in terms of single_hist_sequence
template <typename HistChildT>
void cimg_hist_sequence(cl::sycl::queue &Q, hist::Config Cfg,
                        EBundleTy ExeBundle) {
  using Ty = typename HistChildT::type;
  std::cout << "Initializing with image: " << Cfg.Image << std::endl;
  cimg_library::CImg<unsigned char> Image(Cfg.Image.c_str());
  Cfg.Sz = Image.width() * Image.height();
  std::cout << "Overriding size with image size " << Cfg.Sz << std::endl;
  cimg_library::CImgDisplay MainDisp(Image, "Histogram image source");

  // RGB channels
  std::vector<Ty> DataR(Image.data(), Image.data() + Cfg.Sz);
  std::vector<Ty> DataG(Image.data() + Cfg.Sz, Image.data() + 2 * Cfg.Sz);
  std::vector<Ty> DataB(Image.data() + 2 * Cfg.Sz, Image.data() + 3 * Cfg.Sz);
  auto TesterR =
      single_hist_sequence<HistChildT, Ty>(Q, Cfg, DataR.data(), ExeBundle);
  auto TesterG =
      single_hist_sequence<HistChildT, Ty>(Q, Cfg, DataG.data(), ExeBundle);
  auto TesterB =
      single_hist_sequence<HistChildT, Ty>(Q, Cfg, DataB.data(), ExeBundle);

  auto RMaxIt = std::max_element(TesterR.beginBins(), TesterR.endBins());
  auto GMaxIt = std::max_element(TesterG.beginBins(), TesterG.endBins());
  auto BMaxIt = std::max_element(TesterB.beginBins(), TesterB.endBins());
  if (RMaxIt == TesterR.endBins() || GMaxIt == TesterG.endBins() ||
      BMaxIt == TesterB.endBins())
    throw std::runtime_error("Empty bins");

  constexpr int DispHeight = 400;
  cimg_library::CImgDisplay RDisp(Cfg.HistSz * Cfg.BWidth, DispHeight,
                                  "Histogram red channel", 0);
  cimg_library::CImgDisplay GDisp(Cfg.HistSz * Cfg.BWidth, DispHeight,
                                  "Histogram green channel", 0);
  cimg_library::CImgDisplay BDisp(Cfg.HistSz * Cfg.BWidth, DispHeight,
                                  "Histogram blue channel", 0);
  drawer::disp_buffer(RDisp, TesterR.dataBins(), TesterR.sizeBins(), *RMaxIt,
                      drawer::red);
  drawer::disp_buffer(GDisp, TesterG.dataBins(), TesterG.sizeBins(), *GMaxIt,
                      drawer::green);
  drawer::disp_buffer(BDisp, TesterB.dataBins(), TesterB.sizeBins(), *BMaxIt,
                      drawer::blue);
  while (!MainDisp.is_closed()) {
    cimg_library::cimg::wait(20);
  }
}

template <typename HistChildT>
void test_sequence(int argc, char **argv, sycl::kernel_id kid) {
  using Ty = typename HistChildT::type;
  std::cout << "Welcome to histogram" << std::endl;

  try {
    auto Cfg = hist::read_config(argc, argv);
    auto Q = set_queue();
    print_info(std::cout, Q.get_device());

    IBundleTy SrcBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Q.get_context(), {kid});
    // here we can do specialization constants and many more
    OBundleTy ObjBundle = sycl::compile(SrcBundle);
    EBundleTy ExeBundle = sycl::link(ObjBundle);

    if (Cfg.Image.empty()) {
      std::vector<Ty> Data;
      std::cout << "Initializing with random" << std::endl;
      Data.resize(Cfg.Sz);
      if (Cfg.Zero)
        std::fill(Data.begin(), Data.end(), 0);
      else
        rand_initialize(Data.begin(), Data.end(), 0, Cfg.HistSz - 1);
      single_hist_sequence<HistChildT>(Q, Cfg, Data.data(), ExeBundle);
    } else {
#ifdef CIMG_ENABLE
      cimg_hist_sequence<HistChildT>(Q, Cfg, ExeBundle);
#else
      std::cerr << "Please build with CImg support" << std::endl;
      std::terminate();
#endif // CIMG_ENABLE
    }

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
