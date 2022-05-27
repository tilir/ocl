//------------------------------------------------------------------------------
//
// Generic code to test different variants of image filtering
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//
// Options to control things:
// -img=<image> : path to image, mandatory
// -filt=<filter> : path to filter, mandatory
// -lsz=<l> : local iteration space
// -detailed : detailed report from event
//
// Filter format:
// N, K, x1, x2, .... xN*N
// K is normalization value, xi = xi / K
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
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>

#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#include "testers.hpp"

#ifdef CIMG_ENABLE
#include "drawer.hpp"
#else
#error "Cimg required for this header"
#endif

namespace sycltesters {

// we have sycltesters -> Config in multiple headers,
// so this namespace is required in each.
namespace filter {

constexpr int DEF_LSZ = 32;
constexpr int DEF_DETAILED = 0;

struct Config {
  bool Detailed;
  int LocSz;
  std::string ImagePath, FilterPath;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<std::string>("img", "", "image to apply filter");
  OptParser.template add<std::string>("filt", "", "filter to apply");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.parse(argc, argv);

  Cfg.ImagePath = OptParser.template get<std::string>("img");
  Cfg.FilterPath = OptParser.template get<std::string>("filt");
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.Detailed = OptParser.exists("detailed");

  if (Cfg.LocSz < 1)
    throw std::runtime_error("Wrong parameters (expect all sizes >= 1)");

  if (Cfg.ImagePath.empty())
    throw std::runtime_error("You need to specify image");

  std::cout << "Image path: " << Cfg.ImagePath << std::endl;
  std::cout << "Local size: " << Cfg.LocSz << std::endl;
  if (Cfg.Detailed)
    std::cout << "Detailed events" << std::endl;
  std::cout.flush();
  return Cfg;
}
} // namespace filter

class Filter {
  sycl::queue DeviceQueue_;
  EBundleTy ExeBundle_;

public:
  Filter(sycl::queue &DeviceQueue, EBundleTy ExeBundle)
      : DeviceQueue_(DeviceQueue), ExeBundle_(ExeBundle) {}
  virtual EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                              int ImW, int ImH, drawer::Filter &Filt) = 0;
  sycl::queue &Queue() { return DeviceQueue_; }
  EBundleTy Bundle() const { return ExeBundle_; }
  const sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~Filter() {}
};

struct FilterHost : public Filter {
  FilterHost(sycl::queue &DeviceQueue, EBundleTy ExeBundle)
      : Filter(DeviceQueue, ExeBundle) {}
  EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData, int ImW,
                      int ImH, drawer::Filter &Filt) override {
    std::copy(SrcData, SrcData + ImW * ImH, DstData);
    return {}; // nothing to construct as event
  }
};

class FilterTester {
  Filter &Filter_;
  filter::Config Cfg_;

public:
  FilterTester(Filter &Reduce, filter::Config Cfg)
      : Filter_(Reduce), Cfg_(Cfg) {}

  std::pair<unsigned, unsigned> calculate(sycl::float4 *DstData,
                                          sycl::float4 *SrcData, int ImW,
                                          int ImH, drawer::Filter &Filt) {
    Timer Tm;
    Tm.start();
    EvtRet_t Ret = Filter_(DstData, SrcData, ImW, ImH, Filt);
    Tm.stop();
    auto EvtTiming = getTime(Ret, Cfg_.Detailed ? false : true);
    return {Tm.elapsed(), EvtTiming};
  }
};

constexpr auto sycl_rgba = sycl::image_channel_order::rgba;
constexpr auto sycl_fp32 = sycl::image_channel_type::fp32;

template <typename FilterChildT>
void single_filter_sequence(sycl::queue &Q, filter::Config Cfg,
                            cimg_library::CImgDisplay &Display,
                            sycl::float4 *SrcData, int ImW, int ImH,
                            drawer::Filter &Filt, EBundleTy ExeBundle) {

#if defined(MEASURE_NORMAL)
  std::cout << "Calculating host" << std::endl;
  FilterHost FilterHost{Q, ExeBundle}; // both args here unused
  FilterTester TesterH{FilterHost, Cfg};

#if 0
  std::vector<sycl::float4> DstBufferH(ImW * ImH);
  auto ElapsedH = TesterH.calculate(DstBufferH.data(), SrcData, ImW, ImH, Filt);
  std::cout << "Measured host time: " << ElapsedH.first / msec_per_sec
            << std::endl;
#endif
#endif

  FilterChildT FilterGPU{Q, ExeBundle, Cfg};

  FilterTester Tester{FilterGPU, Cfg};

  std::cout << "Calculating gpu" << std::endl;
#if 1
  std::vector<sycl::float4> DstBufferG(ImW * ImH);
  auto Elapsed = Tester.calculate(DstBufferG.data(), SrcData, ImW, ImH, Filt);
#else
  ImW = 3;
  ImH = 4;
  std::vector<sycl::float4> SrcBufferG(ImW * ImH);
  std::vector<sycl::float4> DstBufferG(ImW * ImH);
  for (int i = 0; i < ImW * ImH; ++i)
    SrcBufferG[i][0] = i;

  std::cout << "before:\n";
  for (int i = 0; i < ImH; ++i) {
    for (int j = 0; j < ImW; ++j)
      std::cout << SrcBufferG[i * ImW + j][0] << " ";
    std::cout << "\n";
  }
  auto Elapsed =
      Tester.calculate(DstBufferG.data(), SrcBufferG.data(), ImW, ImH, Filt);
  std::cout << "after:\n";
  for (int i = 0; i < ImH; ++i) {
    for (int j = 0; j < ImW; ++j)
      std::cout << DstBufferG[i * ImW + j][0] << " ";
    std::cout << "\n";
  }
  return;
#endif

  std::cout << "Measured time: " << Elapsed.first / msec_per_sec << std::endl
            << "Pure execution time: " << Elapsed.second / nsec_per_sec
            << std::endl;

  cimg_library::CImg<unsigned char> Img(ImW, ImH, 1, 3, 255);
  drawer::float4_to_img(DstBufferG.data(), Img);
  Display.display(Img);

#if defined(MEASURE_NORMAL) && defined(VERIFY)
#if 0
  // std::mismatch not working (!)
  auto MisPoint = std::mismatch(DstBufferH.begin(), DstBufferH.end(), DstBufferG.begin());
  if (MisPoint.first != DstBufferH.end()) {
    ptrdiff_t I = std::distance(DstBufferH.begin(), MisPoint.first);
    std::cout << "Mismatch at: " << I << std::endl;    
    throw std::runtime_error("Mismath");
  }
#endif
#endif
}

template <typename FilterChildT>
void test_sequence(int argc, char **argv, sycl::kernel_id kid) {
  std::cout << "Welcome to imaghe filtering" << std::endl;

  try {
    auto Cfg = filter::read_config(argc, argv);
    auto Q = set_queue();
    print_info(std::cout, Q.get_device());

    IBundleTy SrcBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Q.get_context(), {kid});
    // here we can do specialization constants and many more
    OBundleTy ObjBundle = sycl::compile(SrcBundle);
    EBundleTy ExeBundle = sycl::link(ObjBundle);

    std::cout << "Initializing with image: " << Cfg.ImagePath << std::endl;
    cimg_library::CImg<unsigned char> Image(Cfg.ImagePath.c_str());
    const auto ImW = Image.width();
    const auto ImH = Image.height();
    std::cout << "Range: " << ImW << " x " << ImH << std::endl;
#if 1
    cimg_library::CImgDisplay MainDisp(Image, "Filtering image source");
#endif
    cimg_library::CImgDisplay ResultDisp(ImW, ImH, "Filtering image result", 0);

    std::vector<sycl::float4> SrcBuffer(ImW * ImH);
    drawer::img_to_float4(Image, SrcBuffer.data());

    std::cout << "Reading filter: " << Cfg.FilterPath << std::endl;
    drawer::Filter Filt(Cfg.FilterPath);
    std::cout << "N = " << Filt.sqrt_size() << std::endl;

    single_filter_sequence<FilterChildT>(Q, Cfg, ResultDisp, SrcBuffer.data(),
                                         ImW, ImH, Filt, ExeBundle);
#if 1
    while (!MainDisp.is_closed())
      cimg_library::cimg::wait(20);
#endif
  } catch (sycl::exception const &err) {
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
