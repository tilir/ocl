//------------------------------------------------------------------------------
//
// Generic code to test different variants of image boolmachineing
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//
// Options to control things:
// -randboxes=<sz> : random boxes image sz x sz
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

constexpr int DEF_IMSZ = 256;
constexpr int DEF_BM = 0;
constexpr int DEF_LSZ = 16;
constexpr int DEF_DETAILED = 0;
constexpr int DEF_NOVIS = 0;
constexpr int DEF_QUIET = 0;

// number of random boxes
constexpr int NBOXES = 30;

struct Config {
  bool RandMachine, Detailed, Visualize, Quiet, LocOverflow = false;
  int RandImSz, LocSz;
  std::string BoolMachinePath;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<int>("randboxes", DEF_IMSZ, "random boxes image");
  OptParser.template add<std::string>("machine", "", "boolmachine to apply");
  OptParser.template add<int>("randmachine", DEF_BM,
                              "generate random boolmachine");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.template add<int>("novis", DEF_NOVIS,
                              "disable graphic visualization");
  OptParser.template add<int>("quiet", DEF_QUIET, "quiet mode for bulk runs");
  OptParser.parse(argc, argv);

  Cfg.RandImSz = OptParser.template get<int>("randboxes");
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
  qout << "Generating Image with random boxes\n";
  ImageTy Image(Cfg.RandImSz, Cfg.RandImSz, 1, 1, 255);
  drawer::random_boxes_mono(boolmachine::NBOXES, Image, Image.height(),
                            Image.width(), drawer::ItemPatternTy::OutLined);
  return Image;
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
  std::vector<MachineCellTy> DstBuffer_;

public:
  BoolMachineTester(BoolMachine &BM, boolmachine::Config Cfg, int ImW, int ImH)
      : BoolMachine_(BM), Cfg_(Cfg), ImW_(ImW), ImH_(ImH),
        DstBuffer_(ImW * ImH) {}

  using CalcDataTy = std::pair<unsigned, unsigned long long>;
  CalcDataTy calculate(MachineCellTy *SrcData, boolmachine::BoolMachineTy &BM) {
    Timer Tm;
    Tm.start();
    EvtRet_t Ret = BoolMachine_(DstBuffer_.data(), SrcData, ImW_, ImH_, BM);
    Tm.stop();
    auto EvtTiming = getTime(Ret, Cfg_.Detailed ? false : true);
    return {Tm.elapsed(), EvtTiming};
  }

  auto begin() { return DstBuffer_.begin(); }
  auto end() { return DstBuffer_.end(); }
  MachineCellTy *data() { return DstBuffer_.data(); }
};

template <typename Ty>
void disp_tester(const Ty *Data, int ImW, int ImH,
                 cimg_library::CImgDisplay &Disp) {
  ImageTy ResImg(ImW, ImH, 1, 1, 255); // W x H x 1 with 1 color depth
  drawer::scalar_to_img<Ty>(Data, ResImg);
  Disp.display(ResImg);
}

template <typename BoolMachineChildT>
void test_sequence(int argc, char **argv) {
  try {
    auto Cfg = boolmachine::read_config(argc, argv);
    qout << "Welcome to image boolmachineing!\n";
    dump_config_info(Cfg);
    auto Q = set_queue();
    print_info(qout, Q.get_device());

    auto Image = boolmachine::init_image(Cfg);
    const auto ImW = Image.width();
    const auto ImH = Image.height();
    qout << "Range: " << ImW << " x " << ImH << "\n";
    std::vector<MachineCellTy> SrcBuffer(ImW * ImH);
    auto *SrcData = SrcBuffer.data();
    drawer::img_to_scalar<MachineCellTy>(Image, SrcData);
    boolmachine::BoolMachineTy BM = boolmachine::init_boolmachine(Cfg);

    boolmachine::check_device_props(Q.get_device(), Cfg, BM);

#if defined(MEASURE_NORMAL)
    qout << "Calculating host\n";
    BoolMachineHost BoolMachineHost{Q}; // arg unused
    BoolMachineTester TesterH{BoolMachineHost, Cfg, ImW, ImH};
    auto ElapsedH = TesterH.calculate(SrcData, BM);
    qout << "Measured host time: " << ElapsedH.first / msec_per_sec << "\n";
#endif

    BoolMachineChildT BoolMachineGPU{Q, Cfg};
    BoolMachineTester Tester{BoolMachineGPU, Cfg, ImW, ImH};
    qout << "Calculating GPU\n";
    auto Elapsed = Tester.calculate(SrcData, BM);
    qout << "Measured time: " << Elapsed.first / msec_per_sec << "\n";
    auto ExecTime = Elapsed.second / nsec_per_sec;
    qout << "Pure execution time: " << ExecTime << "\n";

    // Quiet mode output: boolmachine size, elapsed time
    if (Cfg.Quiet) {
      qout.set(!Cfg.Quiet);
      qout << ImW << " " << ExecTime << "\n";
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
      cimg_library::CImgDisplay ResDispH(ImW, ImH, "BoolMachine host result",
                                         0);
      disp_tester(TesterH.data(), ImW, ImH, ResDispH);
#endif
      cimg_library::CImgDisplay ResDisp(ImW, ImH, "BoolMachine image result",
                                        0);
      disp_tester(Tester.data(), ImW, ImH, ResDisp);
      while (!ResDisp.is_closed()) {
        cimg_library::cimg::wait(20);
        if (ResDisp.is_keyARROWDOWN()) {
          // TODO: this is too naive, make double-buffering!
          std::copy(Tester.begin(), Tester.end(), SrcBuffer.begin());
          Tester.calculate(SrcBuffer.data(), BM);
          disp_tester(Tester.data(), ImW, ImH, ResDisp);
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
