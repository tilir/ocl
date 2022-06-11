//------------------------------------------------------------------------------
//
// Generic code to test different variants of image rotateing
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//
// Options to control things:
// -img=<image> : path to image
// -randboxes=<sz> : random boxes image sz x sz
// -amt=<deg> : rotate degree
// -lsz=<l> : local iteration space
// -novis : switch off visualization
// -quiet : measurement (quiet) mode
// -detailed : detailed report from event
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
#include <numbers>
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
namespace rotate {

constexpr int DEF_IMSZ = 256;
constexpr int DEF_AMT = 20;
constexpr int DEF_LSZ = 16;
constexpr int DEF_NOVIS = 0;
constexpr int DEF_QUIET = 0;
constexpr int DEF_DETAILED = 0;

// number of random boxes
constexpr int NBOXES = 10;

struct Config {
  bool RandImage, Detailed, Visualize, Quiet;
  int RandImSz, LocSz;
  float Theta;
  std::string ImagePath;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<std::string>("img", "", "image to apply rotate");
  OptParser.template add<int>("randboxes", DEF_IMSZ, "random boxes image");
  OptParser.template add<int>("amt", DEF_AMT, "rotate amount (degrees)");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
  OptParser.template add<int>("novis", DEF_NOVIS,
                              "disable graphic visualization");
  OptParser.template add<int>("quiet", DEF_QUIET, "quiet mode for bulk runs");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.parse(argc, argv);

  Cfg.ImagePath = OptParser.template get<std::string>("img");
  Cfg.RandImage = OptParser.exists("randboxes");
  Cfg.RandImSz = OptParser.template get<int>("randboxes");
  float Amt = OptParser.template get<int>("amt");
  Cfg.Theta = Amt * std::numbers::pi / 180.0;
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.Visualize = !OptParser.exists("novis");
  Cfg.Quiet = OptParser.exists("quiet");
  Cfg.Detailed = OptParser.exists("detailed");
  if (Cfg.Quiet) {
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
  if (!Cfg.RandImage)
    qout << "Image path: " << Cfg.ImagePath << "\n";
  else
    qout << "Random image, size = " << Cfg.RandImSz << "\n";
  qout << "Local size: " << Cfg.LocSz << "\n";
  qout << "Rotating angle (radians): " << Cfg.Theta << "\n";
  if (Cfg.Detailed)
    qout << "Detailed events\n";
  if (Cfg.Visualize)
    qout << "Screen visualization\n";
}

inline void check_device_props(sycl::device D, Config &Cfg) {
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
}

inline ImageTy init_image(rotate::Config Cfg) {
  if (!Cfg.ImagePath.empty()) {
    qout << "Initializing with image: " << Cfg.ImagePath << "\n";
    ImageTy Image(Cfg.ImagePath.c_str());
    return Image;
  }

  if (Cfg.RandImage) {
    qout << "Generating Image with random boxes\n";
    ImageTy Image(Cfg.RandImSz, Cfg.RandImSz, 1, 3, 255);
    drawer::random_boxes(rotate::NBOXES, Image);
    return Image;
  }

  throw std::runtime_error("Need image");
}

} // namespace rotate

class Rotate {
  sycl::queue DeviceQueue_;

public:
  Rotate(sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                              int ImW, int ImH, float Theta) = 0;
  sycl::queue &Queue() { return DeviceQueue_; }
  const sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~Rotate() {}
};

struct RotateHost : public Rotate {
  RotateHost(sycl::queue &DeviceQueue) : Rotate(DeviceQueue) {}
  EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData, int ImW,
                      int ImH, float Theta) override {
    std::fill(DstData, DstData + ImW * ImH, sycl::float4{0, 0, 0, 0});
    float X0 = ImW / 2.0f;
    float Y0 = ImH / 2.0f;

    qout << "X0 = " << X0 << ", Y0 = " << Y0 << "\n";

    for (int Row = 0; Row < ImH; ++Row) {
      for (int Column = 0; Column < ImW; ++Column) {
        float Xprime = Column - X0;
        float Yprime = Row - Y0;

        int Xr = Xprime * cos(Theta) - Yprime * sin(Theta) + X0;
        int Yr = Xprime * sin(Theta) + Yprime * cos(Theta) + Y0;

        if ((Xr < ImW) && (Yr < ImH) && (Xr >= 0) && (Yr >= 0))
          DstData[Row * ImW + Column] = SrcData[Yr * ImW + Xr];
      }
    }
    return {}; // nothing to construct as event
  }
};

class RotateTester {
  Rotate &Rotate_;
  rotate::Config Cfg_;
  int ImW_, ImH_;
  std::vector<sycl::float4> DstBuffer_;

public:
  RotateTester(Rotate &Rotate, rotate::Config Cfg, int ImW, int ImH)
      : Rotate_(Rotate), Cfg_(Cfg), ImW_(ImW), ImH_(ImH),
        DstBuffer_(ImW * ImH) {}

  using CalcDataTy = std::pair<unsigned, unsigned long long>;
  CalcDataTy calculate(sycl::float4 *SrcData, float Theta) {
    Timer Tm;
    Tm.start();
    EvtRet_t Ret = Rotate_(DstBuffer_.data(), SrcData, ImW_, ImH_, Theta);
    Tm.stop();
    auto EvtTiming = getTime(Ret, Cfg_.Detailed ? false : true);
    return {Tm.elapsed(), EvtTiming};
  }

  auto begin() { return DstBuffer_.begin(); }
  auto end() { return DstBuffer_.end(); }
  auto *data() { return DstBuffer_.data(); }
};

template <typename Ty>
void disp_tester(const Ty *Data, int ImW, int ImH,
                 cimg_library::CImgDisplay &Disp) {
  ImageTy ResImg(ImW, ImH, 1, 3, 255);
  drawer::float4_to_img(Data, ResImg);
  Disp.display(ResImg);
}

template <typename RotateChildT> void test_sequence(int argc, char **argv) {
  try {
    auto Cfg = rotate::read_config(argc, argv);
    qout << "Welcome to image rotateing!\n";
    dump_config_info(Cfg);
    auto Q = set_queue();
    print_info(qout, Q.get_device());

    auto Image = rotate::init_image(Cfg);
    const auto ImW = Image.width();
    const auto ImH = Image.height();
    qout << "Range: " << ImW << " x " << ImH << "\n";
    std::vector<sycl::float4> SrcBuffer(ImW * ImH);
    drawer::img_to_float4(Image, SrcBuffer.data());

    rotate::check_device_props(Q.get_device(), Cfg);

#if defined(MEASURE_NORMAL)
    qout << "Calculating host\n";
    RotateHost RotateHost{Q}; // arg unused
    RotateTester TesterH{RotateHost, Cfg, ImW, ImH};
    auto ElapsedH = TesterH.calculate(SrcBuffer.data(), Cfg.Theta);
    qout << "Measured host time: " << ElapsedH.first / msec_per_sec << "\n";
#endif

    RotateChildT RotateGPU{Q, Cfg};
    RotateTester Tester{RotateGPU, Cfg, ImW, ImH};
    qout << "Calculating GPU\n";
    auto Elapsed = Tester.calculate(SrcBuffer.data(), Cfg.Theta);
    qout << "Measured time: " << Elapsed.first / msec_per_sec << "\n";
    auto ExecTime = Elapsed.second / nsec_per_sec;
    qout << "Pure execution time: " << ExecTime << "\n";

    // Quiet mode output: rotate size, elapsed time
    if (Cfg.Quiet) {
      qout.set(!Cfg.Quiet);
      qout << ImW << " " << ExecTime << "\n";
      qout.set(Cfg.Quiet);
    }

#if defined(MEASURE_NORMAL) && defined(VERIFY)
    // Do we need formal verification here? Is it even possible?
    // What worries me a lot: we are interpolating on GPU with sampler, so
    // pixel-to-pixel we may have different picture, which is fine.
#endif

    if (Cfg.Visualize) {
      cimg_library::CImgDisplay MainDisp(Image, "Rotateing image source");
#if defined(MEASURE_NORMAL)
      cimg_library::CImgDisplay ResDispH(ImW, ImH, "Image result: host", 0);
      disp_tester(TesterH.data(), ImW, ImH, ResDispH);
#endif
      cimg_library::CImgDisplay ResDisp(ImW, ImH, "Image result: GPU", 0);
      disp_tester(Tester.data(), ImW, ImH, ResDisp);
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
