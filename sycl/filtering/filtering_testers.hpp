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

constexpr int DEF_LSZ = 16;
constexpr int DEF_DETAILED = 0;
constexpr int DEF_IMSZ = 256;
constexpr int DEF_FILTSZ = 3;
constexpr int DEF_QUIET = 0;
constexpr int DEF_NOVIS = 0;

// min/max for random filter before normalization
constexpr int MINVAL = -16;
constexpr int MAXVAL = 16;

// number of random boxes
constexpr int NBOXES = 10;

struct Config {
  bool Detailed, RandImage = false, RandFilter = false, Visualize = true,
                 Quiet = false, LocOverflow = false;
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
  OptParser.template add<int>("novis", DEF_NOVIS,
                              "disable graphic visualization");
  OptParser.template add<int>("quiet", DEF_QUIET, "quiet mode for bulk runs");
  OptParser.parse(argc, argv);

  Cfg.ImagePath = OptParser.template get<std::string>("img");
  Cfg.FilterPath = OptParser.template get<std::string>("filt");
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.Detailed = OptParser.exists("detailed");
  Cfg.RandFilter = OptParser.exists("randfilter");
  Cfg.RandFiltSz = OptParser.template get<int>("randfilter");
  Cfg.RandImage = OptParser.exists("randboxes");
  Cfg.RandImSz = OptParser.template get<int>("randboxes");
  if (OptParser.exists("novis"))
    Cfg.Visualize = false;
  if (OptParser.exists("quiet")) {
    Cfg.Quiet = true;
    Cfg.Visualize = false; // quiet implies novis of course
    qout.set(Cfg.Quiet);
  }
  return Cfg;
}

inline void dump_config_info(Config &Cfg) {
  if (Cfg.LocSz < 1)
    throw std::runtime_error("Wrong parameters (expect all sizes >= 1)");

  if (!Cfg.RandImage && Cfg.ImagePath.empty())
    throw std::runtime_error("You need to specify image");

  if (!Cfg.RandFilter && Cfg.FilterPath.empty())
    throw std::runtime_error("You need to specify filter");

  qout << "Image path: " << Cfg.ImagePath << "\n";
  qout << "Local size: " << Cfg.LocSz << "\n";
  if (Cfg.Detailed)
    qout << "Detailed events\n";
  if (Cfg.Visualize)
    qout << "Screen visualization\n";
}

inline void check_device_props(sycl::device D, Config &Cfg,
                               drawer::Filter &Filt) {
  if (!D.has(sycl::aspect::image))
    throw std::runtime_error("Image support required");

  constexpr auto max_lsz = sycl::info::device::max_work_group_size;
  const auto MLSZ = D.template get_info<max_lsz>();
  qout << "Max WG size: " << MLSZ << std::endl;

  if (Cfg.LocSz * Cfg.LocSz > MLSZ)
    throw std::runtime_error("Local work group of this size not supported");

  constexpr auto max_gmem = sycl::info::device::global_mem_size;
  constexpr int bytes_to_kilobytes = 1024;
  const auto MaxGMEM = D.template get_info<max_gmem>() / bytes_to_kilobytes;
  qout << "Max global mem size: " << MaxGMEM << " kbytes" << std::endl;

  constexpr auto max_lmem = sycl::info::device::local_mem_size;
  const auto MaxLMEM = D.template get_info<max_lmem>();
  qout << "Max local mem size: " << MaxLMEM << " bytes" << std::endl;

  int FiltSize = Filt.sqrt_size();
  int HalfWidth = FiltSize / 2;
  int LocMem = Cfg.LocSz + HalfWidth * 2;
  int ReqMem = LocMem * LocMem * sizeof(sycl::float4);
  if (ReqMem > MaxLMEM) {
    qout << "Warning: " << ReqMem << " bytes of local memory will be required "
         << "which exceeds max local memory" << std::endl;
    Cfg.LocOverflow = 1;
  }
}

} // namespace filter

class Filter {
  sycl::queue DeviceQueue_;

public:
  Filter(sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                              int ImW, int ImH, drawer::Filter &Filt) = 0;
  sycl::queue &Queue() { return DeviceQueue_; }
  const sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~Filter() {}
};

struct FilterHost : public Filter {
  FilterHost(sycl::queue &DeviceQueue) : Filter(DeviceQueue) {}
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
            sycl::float4 Pixel = {0.0f, 0.0f, 0.0f, 0.0f};
            int X = Column + J;
            int Y = Row + I;
            if (X < 0)
              X = 0;
            else if (X > ImW - 1)
              X = ImW - 1;
            else if (Y < 0)
              Y = 0;
            else if (Y > ImH - 1)
              Y = ImH - 1;
            else
              Pixel = SrcData[Y * ImW + X];
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

  using CalcDataTy = std::pair<unsigned, unsigned long long>;
  CalcDataTy calculate(sycl::float4 *SrcData, drawer::Filter &Filt) {
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
                                    drawer::Filter &Filt) {

#if defined(MEASURE_NORMAL)
  qout << "Calculating host\n";
  FilterHost FilterHost{Q}; // arg unused
  FilterTester TesterH{FilterHost, Cfg, ImW, ImH};
  auto ElapsedH = TesterH.calculate(SrcData, Filt);
  qout << "Measured host time: " << ElapsedH.first / msec_per_sec << "\n";
#endif

  FilterChildT FilterGPU{Q, Cfg};
  FilterTester Tester{FilterGPU, Cfg, ImW, ImH};
  qout << "Calculating GPU\n";
  auto Elapsed = Tester.calculate(SrcData, Filt);
  qout << "Measured time: " << Elapsed.first / msec_per_sec << "\n";
  auto ExecTime = Elapsed.second / nsec_per_sec;
  qout << "Pure execution time: " << ExecTime << "\n";

  // Quiet mode output: filter size, elapsed time
  if (Cfg.Quiet) {
    qout.set(!Cfg.Quiet);
    qout << ImW << " " << ExecTime << "\n";
    qout.set(Cfg.Quiet);
  }

#if defined(MEASURE_NORMAL) && defined(VERIFY)
  auto *DataH = TesterH.data();
  auto *DataG = Tester.data();
  for (int Row = 0; Row < ImH; ++Row)
    for (int Column = 0; Column < ImW; ++Column) {
      sycl::float4 H = DataH[Row * ImW + Column];
      sycl::float4 G = DataG[Row * ImW + Column];
      if (!sycl::all(H - G < 0.0001f)) {
        qout << "Mismatch at: (" << Column << ", " << Row << ")\n";
        qout << "Host: [" << H[0] << ", " << H[1] << ", " << H[2] << ", "
             << H[3] << "]\n";
        qout << "GPU: [" << G[0] << ", " << G[1] << ", " << G[2] << ", " << G[3]
             << "]\n";
        throw std::runtime_error("Mismath");
      }
    }
#endif
  return Tester;
}

inline drawer::Filter init_filter(filter::Config Cfg) {
  if (!Cfg.FilterPath.empty()) {
    qout << "Reading filter: " << Cfg.FilterPath << "\n";
    drawer::Filter Filt(Cfg.FilterPath);
    qout << "N = " << Filt.sqrt_size() << "\n";
    return Filt;
  }

  if (Cfg.RandFilter) {
    qout << "Generating filter\n";
    drawer::Filter Filt(Cfg.RandFiltSz, filter::MINVAL, filter::MAXVAL);
    qout << "N = " << Filt.sqrt_size() << "\n";
#ifdef SHOWFILT
    auto DataSz = Cfg.RandFiltSz * Cfg.RandFiltSz;
    sycltesters::visualize_seq(Filt.data(), Filt.data() + DataSz, std::cout);
#endif
    return Filt;
  }

  throw std::runtime_error("Need filter");
}

using ImageTy = cimg_library::CImg<unsigned char>;

inline ImageTy init_image(filter::Config Cfg) {
  if (!Cfg.ImagePath.empty()) {
    qout << "Initializing with image: " << Cfg.ImagePath << "\n";
    ImageTy Image(Cfg.ImagePath.c_str());
    return Image;
  }

  if (Cfg.RandImage) {
    qout << "Generating Image with random boxes\n";
    ImageTy Image(Cfg.RandImSz, Cfg.RandImSz, 1, 3, 255);
    drawer::random_boxes(filter::NBOXES, Image);
    return Image;
  }

  throw std::runtime_error("Need image");
}

template <typename FilterChildT> void test_sequence(int argc, char **argv) {
  try {
    auto Cfg = filter::read_config(argc, argv);
    qout << "Welcome to image filtering!\n";
    dump_config_info(Cfg);
    auto Q = set_queue();
    print_info(qout, Q.get_device());

    auto Image = init_image(Cfg);
    const auto ImW = Image.width();
    const auto ImH = Image.height();
    qout << "Range: " << ImW << " x " << ImH << "\n";
    std::vector<sycl::float4> SrcBuffer(ImW * ImH);
    drawer::img_to_float4(Image, SrcBuffer.data());
    drawer::Filter Filt = init_filter(Cfg);

    filter::check_device_props(Q.get_device(), Cfg, Filt);

    auto Tester = single_filter_sequence<FilterChildT>(Q, Cfg, SrcBuffer.data(),
                                                       ImW, ImH, Filt);

    // display source and result pictures
    if (Cfg.Visualize) {
      cimg_library::CImgDisplay MainDisp(Image, "Filtering image source");
      cimg_library::CImgDisplay ResDisp(ImW, ImH, "Filtering image result", 0);

      ImageTy ResImg(ImW, ImH, 1, 3, 255);
      drawer::float4_to_img(Tester.data(), ResImg);
      ResDisp.display(ResImg);

      while (!MainDisp.is_closed())
        cimg_library::cimg::wait(20);
    }
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
  qout << "Everything is correct\n";
}

} // namespace sycltesters
