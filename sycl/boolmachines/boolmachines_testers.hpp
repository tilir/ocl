//------------------------------------------------------------------------------
//
// Generic code to test different variants of image boolmachineing
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//
// Options to control things:
// -imw=<width>
// -imh=<height>
// -randboxes : random boxes image imw x imh
// -randpoints : random points image imw x imh
// -img=<path> : starting image from file
// if boxes/image not specified, start with imw x imh with
// single black dot in the center
//
// -machine=<machine> : path to machine, rand machine will be used otherwise
// -randmachine : random machine 3x3
// -lsz=<l> : local iteration space
// -novis : switch off visualization
// -quiet : measurement (quiet) mode
// -detailed : detailed report from event
//
// Machine format:
// E0 F1 2C ... 13 (64 bytes, 512 bits, 3x3 boolean function)
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

using MachineCellTy = unsigned char;

namespace sycltesters {

// we have sycltesters -> Config in multiple headers,
// so this namespace is required in each.
namespace boolmachine {

constexpr int DEF_IMW = 512;
constexpr int DEF_IMH = 512;
constexpr int DEF_RB = 0;
constexpr int DEF_BM = 0;
constexpr int DEF_LSZ = 16;
constexpr int DEF_DETAILED = 0;
constexpr int DEF_NOVIS = 0;
constexpr int DEF_QUIET = 0;

// number of random boxes
constexpr int NBOXES = 30;

// fast-forward iteration number
constexpr int FF_ITER_COUNT = 300;

enum class Starting {
  SINGLEDOT,
  DOTS,
  BOXES,
  IMAGE
};

struct Config {
  bool RandMachine, Detailed, Visualize, Quiet, LocOverflow = false;
  int LocSz, ImW, ImH;
  std::string BoolMachinePath, ImagePath;
  Starting InitType;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<int>("imw", DEF_IMW, "image width");
  OptParser.template add<int>("imh", DEF_IMH, "image height");
  OptParser.template add<std::string>("img", "", "image file to apply machine or (randboxes | dots | singledot)");

  OptParser.template add<std::string>("machine", "", "boolmachine to apply");
  OptParser.template add<int>("randmachine", DEF_BM,
                              "generate random boolmachine");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.template add<int>("novis", DEF_NOVIS,
                              "disable graphic visualization");
  OptParser.template add<int>("quiet", DEF_QUIET, "quiet mode for bulk runs");
  OptParser.parse(argc, argv);

  Cfg.ImW = OptParser.template get<int>("imw");
  Cfg.ImH = OptParser.template get<int>("imh");
  Cfg.ImagePath = OptParser.template get<std::string>("img");

  Cfg.InitType = Starting::IMAGE;
  if (Cfg.ImagePath == "randboxes" || Cfg.ImagePath == "boxes")
    Cfg.InitType = Starting::BOXES;
  if (Cfg.ImagePath == "dots")
    Cfg.InitType = Starting::DOTS;
  if (Cfg.ImagePath == "singledot")
    Cfg.InitType = Starting::SINGLEDOT;

  Cfg.BoolMachinePath = OptParser.template get<std::string>("machine");
  Cfg.RandMachine =
      OptParser.exists("randmachine") || Cfg.BoolMachinePath.empty();
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.Detailed = OptParser.exists("detailed");
  Cfg.Visualize = !OptParser.exists("novis");
  Cfg.Quiet = OptParser.exists("quiet");

  if (Cfg.Quiet) {
    Cfg.Visualize = false; // quiet implies novis of course
    qout.set(Cfg.Quiet);
  }
  return Cfg;
}

inline void dump_config_info(Config &Cfg) {
  if (Cfg.LocSz < 1)
    throw std::runtime_error("Wrong parameters (expect all sizes >= 1)");

  if (!Cfg.RandMachine && Cfg.BoolMachinePath.empty())
    throw std::runtime_error("You need to specify boolmachine");

  if (!Cfg.BoolMachinePath.empty())
    qout << "Machine path: " << Cfg.BoolMachinePath << "\n";
  qout << "Local size: " << Cfg.LocSz << "\n";
  if (Cfg.Detailed)
    qout << "Detailed events\n";
  if (Cfg.Visualize)
    qout << "Screen visualization\n";
}

class BoolMachineTy {
public:
  static constexpr int NELTS = 64;

private:
  // 3 x 3 = 9
  // 1 << (1 << 9) = 512
  // 512 / 8 = 64
  std::array<MachineCellTy, NELTS> Desc;

public:
  BoolMachineTy() {
    sycltesters::Dice D(0, 255);
    std::generate(Desc.begin(), Desc.end(), [&] { return D(); });
  }

  BoolMachineTy(std::string Filepath) {
    std::ifstream Is(Filepath), EndIs;
    Is.exceptions(std::istream::failbit);
    Is >> std::hex;
    std::istream_iterator<int> IsIt{Is};
    std::copy_n(IsIt, NELTS, Desc.begin());
  }

  void dump(std::ostream &Os) const {
    Os << std::hex;
    std::ostream_iterator<int> OsIt{Os, " "};
    std::copy(Desc.begin(), Desc.end(), OsIt);
  }

  MachineCellTy get(int Idx) const {
    assert(Idx < NELTS * CHAR_BIT);
    int ByteIdx = Idx / CHAR_BIT;
    int BitIdx = Idx % CHAR_BIT;
    return (Desc[ByteIdx] >> BitIdx) & 1;
  }

  MachineCellTy *data() { return Desc.data(); }
};

inline void check_device_props(sycl::device D, Config &Cfg, BoolMachineTy &BM) {
  if (!D.has(sycl::aspect::image))
    throw std::runtime_error("Image support required");

  constexpr auto MaxLocSz = sycl::info::device::max_work_group_size;
  const auto MLSZ = D.template get_info<MaxLocSz>();
  qout << "Max WG size: " << MLSZ << std::endl;

  if (Cfg.LocSz * Cfg.LocSz > MLSZ)
    throw std::runtime_error("Local work group of this size not supported");

  constexpr auto MaxGmem = sycl::info::device::global_mem_size;
  constexpr int bytes_to_kilobytes = 1024;
  const auto MaxGlobalMEM = D.template get_info<MaxGmem>() / bytes_to_kilobytes;
  qout << "Max global mem size: " << MaxGlobalMEM << " kbytes" << std::endl;

  constexpr auto MaxLmem = sycl::info::device::local_mem_size;
  const auto MaxLocalMEM = D.template get_info<MaxLmem>();
  qout << "Max local mem size: " << MaxLocalMEM << " bytes" << std::endl;
}

inline BoolMachineTy init_boolmachine(boolmachine::Config Cfg) {
  if (!Cfg.BoolMachinePath.empty()) {
    qout << "Reading boolmachine: " << Cfg.BoolMachinePath << "\n";
    BoolMachineTy BM(Cfg.BoolMachinePath);
    BM.dump(qout);
    qout << std::endl;
    return BM;
  }

  if (Cfg.RandMachine) {
    qout << "Generating boolmachine\n";
    BoolMachineTy BM;
    BM.dump(qout);
    qout << std::endl;
    return BM;
  }

  throw std::runtime_error("Need boolmachine");
}

inline ImageTy init_image(boolmachine::Config Cfg) {
  if (Cfg.InitType == Starting::IMAGE) {
    qout << "Initializing with image: " << Cfg.ImagePath << "\n";
    ImageTy Image(Cfg.ImagePath.c_str());
    return Image;
  }

  if (Cfg.InitType == Starting::BOXES) {
    qout << "Generating Image with random boxes\n";
    ImageTy Image(Cfg.ImW, Cfg.ImH, 1, 1, 255);

    drawer::random_boxes_mono(boolmachine::NBOXES, Image, Image.height(),
                              Image.width(), drawer::ItemPatternTy::OutLined);
    return Image;
  }

  if (Cfg.InitType == Starting::SINGLEDOT) {
    qout << "Generating Image with single central dot\n";
    ImageTy Image(Cfg.ImW, Cfg.ImH, 1, 1, 255);
    unsigned char color[1] = {0};
    Image.draw_point(Cfg.ImW / 2, Cfg.ImH / 2, color);
    return Image;
  }

  if (Cfg.InitType == Starting::DOTS) {
    qout << "For each dot make it uniformly black or white\n";
    ImageTy Image(Cfg.ImW, Cfg.ImH, 1, 1, 255);
    unsigned char color[1] = {0};
    sycltesters::Dice D(0, 1);
    for (int W = 0; W < Cfg.ImW; ++W)
      for (int H = 0; H < Cfg.ImH; ++H)
        if (D())
          Image.draw_point(W, H, color);
    return Image;
  }

  throw std::runtime_error("Need image");
}

} // namespace boolmachine

class BoolMachine {
  sycl::queue DeviceQueue_;

public:
  BoolMachine(sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(MachineCellTy *DstData, MachineCellTy *SrcData,
                              int ImW, int ImH,
                              boolmachine::BoolMachineTy &BM) = 0;
  sycl::queue &Queue() { return DeviceQueue_; }
  const sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~BoolMachine() {}
};

struct BoolMachineHost : public BoolMachine {
  BoolMachineHost(sycl::queue &DeviceQueue) : BoolMachine(DeviceQueue) {}
  EvtRet_t operator()(MachineCellTy *DstData, MachineCellTy *SrcData, int ImW,
                      int ImH, boolmachine::BoolMachineTy &BM) override {
    constexpr int FiltSize = 3;
    constexpr int HalfWidth = FiltSize / 2;

    for (int Row = 0; Row < ImH; ++Row) {
      for (int Column = 0; Column < ImW; ++Column) {
        MachineCellTy Idx = 0;
        int Shift = 0;
        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            MachineCellTy Value = 0;
            int X = Column + J;
            int Y = Row + I;
            if (X >= 0 && X < ImW && Y >= 0 && Y < ImH)
              Value = SrcData[Y * ImW + X];
            Value = Value ? 1 : 0;
            Idx += Value << Shift;
            Shift += 1;
          }
        }
        // Idx is bit number in BM
        DstData[Row * ImW + Column] = BM.get(Idx);
      }
    }

    return {}; // nothing to construct as event
  }
};

class BoolMachineTester {
  BoolMachine &BoolMachine_;
  boolmachine::Config Cfg_;
  int ImW_, ImH_;
  std::vector<MachineCellTy> FstBuffer_, SndBuffer_;
  bool FstActive;

  // ref-to-vector looks ugly but I need resize
  static auto initbm(std::vector<MachineCellTy> &SrcVec,
                     boolmachine::Config Cfg) {
    auto Image = boolmachine::init_image(Cfg);
    int ImW = Image.width();
    int ImH = Image.height();
    SrcVec.resize(ImW * ImH);
    drawer::img_to_scalar<MachineCellTy>(Image, SrcVec.data());
    return std::make_pair(ImW, ImH);
  }

public:
  BoolMachineTester(BoolMachine &BM, boolmachine::Config Cfg)
      : BoolMachine_(BM), Cfg_(Cfg) {
    std::tie(ImW_, ImH_) = initbm(FstBuffer_, Cfg_);
    assert(FstBuffer_.size() == ImW_ * ImH_);
    SndBuffer_.resize(FstBuffer_.size());
    FstActive = true;
  }

  void reinit() {
    std::tie(ImW_, ImH_) = initbm(FstBuffer_, Cfg_);
    SndBuffer_.resize(FstBuffer_.size());
    FstActive = true;
  }

  void disp_dst(cimg_library::CImgDisplay &Disp) {
    unsigned char black[1] = {0};
    unsigned char white[1] = {255};
    MachineCellTy *Dst = FstActive ? SndBuffer_.data() : FstBuffer_.data();
    ImageTy ResImg(ImW_, ImH_, 1, 1, 255); // W x H x 1 with 1 color depth
    drawer::scalar_to_img<MachineCellTy>(Dst, ResImg);
    ResImg.draw_text(ImW_ - 200, ImH_ - 13 * 6, " down : one step of machine ",
                     white, black, /* opacity */ 1, /* font */ 13);
    ResImg.draw_text(ImW_ - 200, ImH_ - 13 * 5, " f : fast forward ", white,
                     black, /* opacity */ 1, /* font */ 13);
    ResImg.draw_text(ImW_ - 200, ImH_ - 13 * 4,
                     " r : reinit machine and image ", white, black,
                     /* opacity */ 1, /* font */ 13);
    ResImg.draw_text(ImW_ - 200, ImH_ - 13 * 3,
                     " i : reinit image, keep machine ", white, black,
                     /* opacity */ 1, /* font */ 13);
    Disp.display(ResImg);
  }

  using CalcDataTy = std::pair<unsigned, unsigned long long>;

  // Double buffering. Next calculation uses buffer from previous one.
  CalcDataTy calculate(boolmachine::BoolMachineTy &BM) {
    MachineCellTy *Dst = FstActive ? SndBuffer_.data() : FstBuffer_.data();
    MachineCellTy *Src = FstActive ? FstBuffer_.data() : SndBuffer_.data();
    FstActive = !FstActive;
    Timer Tm;
    Tm.start();
    EvtRet_t Ret = BoolMachine_(Dst, Src, ImW_, ImH_, BM);
    Tm.stop();
    auto EvtTiming = getTime(Ret, Cfg_.Detailed ? false : true);
    return {Tm.elapsed(), EvtTiming};
  }

  auto begin() { return FstActive ? SndBuffer_.begin() : FstBuffer_.begin(); }
  auto end() { return FstActive ? SndBuffer_.end() : FstBuffer_.end(); }
  MachineCellTy *data() {
    return FstActive ? SndBuffer_.data() : FstBuffer_.data();
  }
  int width() const { return ImW_; }
  int height() const { return ImH_; }
};

template <typename BoolMachineChildT>
void test_sequence(int argc, char **argv) {
  try {
    auto Cfg = boolmachine::read_config(argc, argv);
    qout << "Welcome to image boolmachineing!\n";
    dump_config_info(Cfg);
    auto Q = set_queue();
    print_info(qout, Q.get_device());

    boolmachine::BoolMachineTy BM = boolmachine::init_boolmachine(Cfg);
    boolmachine::check_device_props(Q.get_device(), Cfg, BM);

#if defined(MEASURE_NORMAL)
    qout << "Calculating host\n";
    BoolMachineHost BoolMachineHost{Q}; // arg unused
    BoolMachineTester TesterH{BoolMachineHost, Cfg};
    auto ElapsedH = TesterH.calculate(BM);
    qout << "Measured host time: " << ElapsedH.first / msec_per_sec << "\n";
#endif

    BoolMachineChildT BoolMachineGPU{Q, Cfg};
    BoolMachineTester Tester{BoolMachineGPU, Cfg};
    qout << "Calculating GPU\n";
    auto Elapsed = Tester.calculate(BM);
    qout << "Measured time: " << Elapsed.first / msec_per_sec << "\n";
    auto ExecTime = Elapsed.second / nsec_per_sec;
    qout << "Pure execution time: " << ExecTime << "\n";

    // Quiet mode output: image size, elapsed time
    if (Cfg.Quiet) {
      qout.set(!Cfg.Quiet);
      qout << Tester.width() << " " << ExecTime << "\n";
      qout.set(Cfg.Quiet);
    }

#if defined(MEASURE_NORMAL) && defined(VERIFY)
// TODO: check here
#endif

    if (Cfg.Visualize) {
#if defined(SHOW_ORIG)
      cimg_library::CImgDisplay MainDisp(Image, "BoolMachine image source");
#endif
#if defined(MEASURE_NORMAL)
      cimg_library::CImgDisplay ResDispH(TesterH.width(), TesterH.height(),
                                         "BoolMachine host result", 0);
      TesterH.disp_dst(ResDispH);
#endif
      cimg_library::CImgDisplay ResDisp(Tester.width(), Tester.height(),
                                        "BoolMachine image result", 0);
      Tester.disp_dst(ResDisp);
      while (!ResDisp.is_closed()) {
        cimg_library::cimg::wait(50);
        bool Updated = false;
        // key down for one calcualtion
        if (ResDisp.is_keyARROWDOWN()) {
          Tester.calculate(BM);
          Updated = true;
        }
        // 'r' for bool machine re-generate
        // image also re-inited
        if (ResDisp.is_key(cimg_library::cimg::keyR)) {
          BM = boolmachine::init_boolmachine(Cfg);
          Tester.reinit();
          Tester.calculate(BM);
          Updated = true;
        }
        // 'i' for image reload with same machine
        if (ResDisp.is_key(cimg_library::cimg::keyI)) {
          Tester.reinit();
          Tester.calculate(BM);
          Updated = true;
        }
        // 'f' for 'fast forward'
        if (ResDisp.is_key(cimg_library::cimg::keyF)) {
          for (int I = 0; I < boolmachine::FF_ITER_COUNT; ++I)
            Tester.calculate(BM);
          Updated = true;
        }
        if (Updated) {
          Tester.disp_dst(ResDisp);
          ResDisp.flush();
        }
      }
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
