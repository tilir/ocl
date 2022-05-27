//------------------------------------------------------------------------------
//
// Generic code to test different variants of image filtering
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//
// Options to control things:
// -img=<image> : path to image
// -randboxes=<sz> : random boxes image sz x sz
// -filt=<filter> : path to filter, mandatory
// -randfilter=<sz> : random filter sz x sz
// -lsz=<l> : local iteration space
// -novis : switch off visualization
// -quiet : measurement (quiet) mode
// -detailed : detailed report from event
//
// Filter format:
// N, K, x1, x2, .... xN*N
// K is normalization value, xi = xi / K
//
// try:
// > filtering_sampler.exe -img=favn.jpg -filt=sharpen.filt
// > filtering_sampler.exe -img=favn.jpg -randfilter=3
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

constexpr auto sycl_rgba = sycl::image_channel_order::rgba;
constexpr auto sycl_fp32 = sycl::image_channel_type::fp32;

namespace sycltesters {

// we have sycltesters -> Config in multiple headers,
// so this namespace is required in each.
namespace filter {

constexpr int DEF_LSZ = 32;
constexpr int DEF_DETAILED = 0;
constexpr int DEF_IMSZ = 100;
constexpr int DEF_FILTSZ = 3;

constexpr int MINFILTER = -16;
constexpr int MAXFILTER = 16;

struct Config {
  bool Detailed, RandImage = false, RandFilter = false;
  int LocSz, RandImSz, RandFiltSz;
  std::string ImagePath, FilterPath;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<std::string>("img", "", "image to apply filter");
  OptParser.template add<int>("randboxes", DEF_IMSZ, "random boxes image");
  OptParser.template add<std::string>("filt", "", "filter to apply");
  OptParser.template add<int>("randfilter", DEF_FILTSZ,
                              "random filter (normalized)");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.parse(argc, argv);

  Cfg.ImagePath = OptParser.template get<std::string>("img");
  Cfg.FilterPath = OptParser.template get<std::string>("filt");
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.Detailed = OptParser.exists("detailed");
  Cfg.RandFilter = OptParser.exists("randfilter");
  Cfg.RandFiltSz = OptParser.template get<int>("randfilter");

  if (Cfg.LocSz < 1)
    throw std::runtime_error("Wrong parameters (expect all sizes >= 1)");

  if (!Cfg.RandImage && Cfg.ImagePath.empty())
    throw std::runtime_error("You need to specify image");

  if (!Cfg.RandFilter && Cfg.FilterPath.empty())
    throw std::runtime_error("You need to specify filter");

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
    auto *FiltPtr = Filt.data();
    int FiltSize = Filt.sqrt_size();
    int HalfWidth = FiltSize / 2;
    for (int Row = 0; Row < ImH; ++Row)
      for (int Column = 0; Column < ImW; ++Column) {
        sycl::float4 Sum = {0.0f, 0.0f, 0.0f, 1.0f};
        int FiltIndex = 0;
        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            // int FiltIndex = (I + HalfWidth) * FiltSize + (J + HalfWidth);
            // first changed index is X
            int X = Column + J;
            if (X < 0)
              X = 0;
            if (X > ImW - 1)
              X = ImW - 1;
            int Y = Row + I;
            if (Y < 0)
              Y = 0;
            if (Y > ImH - 1)
              Y = ImH - 1;
            sycl::float4 Pixel = SrcData[Y * ImW + X];
            Sum[0] += Pixel[0] * FiltPtr[FiltIndex];
            Sum[1] += Pixel[1] * FiltPtr[FiltIndex];
            Sum[2] += Pixel[2] * FiltPtr[FiltIndex];
            FiltIndex += 1;
          }
        }
        DstData[Row * ImW + Column] = Sum;
      }
    return {}; // nothing to construct as event
  }
};

class FilterTester {
  Filter &Filter_;
  filter::Config Cfg_;
  int ImW_, ImH_;
  std::vector<sycl::float4> DstBuffer_;

public:
  FilterTester(Filter &Reduce, filter::Config Cfg, int ImW, int ImH)
      : Filter_(Reduce), Cfg_(Cfg), ImW_(ImW), ImH_(ImH),
        DstBuffer_(ImW * ImH) {}

  std::pair<unsigned, unsigned> calculate(sycl::float4 *SrcData,
                                          drawer::Filter &Filt) {
    Timer Tm;
    Tm.start();
    EvtRet_t Ret = Filter_(DstBuffer_.data(), SrcData, ImW_, ImH_, Filt);
    Tm.stop();
    auto EvtTiming = getTime(Ret, Cfg_.Detailed ? false : true);
    return {Tm.elapsed(), EvtTiming};
  }

  sycl::float4 *data() { return DstBuffer_.data(); }
};

template <typename FilterChildT>
FilterTester single_filter_sequence(sycl::queue &Q, filter::Config Cfg,
                                    sycl::float4 *SrcData, int ImW, int ImH,
                                    drawer::Filter &Filt, EBundleTy ExeBundle) {

#if defined(MEASURE_NORMAL)
  std::cout << "Calculating host" << std::endl;
  FilterHost FilterHost{Q, ExeBundle}; // both args here unused
  FilterTester TesterH{FilterHost, Cfg, ImW, ImH};
  auto ElapsedH = TesterH.calculate(SrcData, Filt);
  std::cout << "Measured host time: " << ElapsedH.first / msec_per_sec
            << std::endl;
#endif

  FilterChildT FilterGPU{Q, ExeBundle, Cfg};
  FilterTester Tester{FilterGPU, Cfg, ImW, ImH};
  std::cout << "Calculating gpu" << std::endl;
  auto Elapsed = Tester.calculate(SrcData, Filt);
  std::cout << "Measured time: " << Elapsed.first / msec_per_sec << std::endl
            << "Pure execution time: " << Elapsed.second / nsec_per_sec
            << std::endl;

#if defined(MEASURE_NORMAL) && defined(VERIFY)
#if 0 // yet not perfect
  auto *DataH = TesterH.data();
  auto *DataG = Tester.data();
  for (int Row = 0; Row < ImH; ++Row)
    for (int Column = 0; Column < ImW; ++Column) {
      sycl::float4 H = DataH[Row * ImW + Column];
      sycl::float4 G = DataG[Row * ImW + Column];
      if (!sycl::all(H == G)) {
        std::cout << "Mismatch at: (" << Column << ", " << Row << ")"
                  << std::endl;
        std::cout << "Host: [" << H[0] << ", " << H[1] << ", " << H[2] << ", "
                  << H[3] << "]" << std::endl;
        std::cout << "GPU: [" << G[0] << ", " << G[1] << ", " << G[2] << ", "
                  << G[3] << "]" << std::endl;
        throw std::runtime_error("Mismath");
      }
    }
#endif
#endif
  return Tester;
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

    std::vector<sycl::float4> SrcBuffer(ImW * ImH);
    drawer::img_to_float4(Image, SrcBuffer.data());

    drawer::Filter Filt;
    if (!Cfg.FilterPath.empty()) {
      std::cout << "Reading filter: " << Cfg.FilterPath << std::endl;
      Filt = drawer::Filter(Cfg.FilterPath);
      std::cout << "N = " << Filt.sqrt_size() << std::endl;
    } else if (Cfg.RandFilter) {
      std::cout << "Generating filter" << std::endl;
      Filt =
          drawer::Filter(Cfg.RandFiltSz, filter::MINFILTER, filter::MAXFILTER);
      std::cout << "N = " << Filt.sqrt_size() << std::endl;
      sycltesters::visualize_seq(Filt.data(),
                                 Filt.data() + Cfg.RandFiltSz * Cfg.RandFiltSz,
                                 std::cout);
    }

    auto Tester = single_filter_sequence<FilterChildT>(
        Q, Cfg, SrcBuffer.data(), ImW, ImH, Filt, ExeBundle);

    cimg_library::CImgDisplay MainDisp(Image, "Filtering image source");
    cimg_library::CImgDisplay ResultDisp(ImW, ImH, "Filtering image result", 0);

    cimg_library::CImg<unsigned char> ResImg(ImW, ImH, 1, 3, 255);
    drawer::float4_to_img(Tester.data(), ResImg);
    ResultDisp.display(ResImg);

    while (!MainDisp.is_closed())
      cimg_library::cimg::wait(20);
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
