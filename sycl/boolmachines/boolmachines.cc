//------------------------------------------------------------------------------
//
// BoolMachineing with 2D convolution kernel (SYCL vs serial CPU)
// Demonstrates samplers
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <cassert>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "boolmachines_testers.hpp"

using ConfigTy = sycltesters::boolmachine::Config;
using BoolMachineTy = sycltesters::boolmachine::BoolMachineTy;

// class is used for kernel name
class boolmachine_2d_buf;

class BoolMachineBuffer : public sycltesters::BoolMachine {
  using sycltesters::BoolMachine::Queue;
  ConfigTy Cfg_;

public:
  BoolMachineBuffer(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::BoolMachine(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(MachineCellTy *DstData,
                                   MachineCellTy *SrcData, int ImW, int ImH,
                                   BoolMachineTy &BM) override {
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    sycl::range<2> Dims(ImW, ImH);
    sycl::buffer<MachineCellTy, 2> Dst(DstData, Dims);
    sycl::buffer<MachineCellTy, 2> Src(SrcData, Dims);
    Src.set_final_data(nullptr);
    constexpr int FiltSize = 3;
    constexpr int DataSize = FiltSize * FiltSize;
    constexpr int HalfWidth = FiltSize / 2;

    sycl::buffer<MachineCellTy, 1> BufBM(BoolMachineTy::NELTS);
    {
      auto FiltHostAcc = BufBM.template get_access<sycl_write>();
      const MachineCellTy *BMData = BM.data();
      // TODO: copy BM data to accessor
    }

    // explicit accessor types
    using ImReadTy = sycl::accessor<MachineCellTy, 2, sycl_read, sycl_global>;
    using ImWriteTy = sycl::accessor<MachineCellTy, 2, sycl_write, sycl_global>;

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      ImReadTy InPtr(Src, Cgh);
      ImWriteTy OutPtr(Dst, Cgh);
      auto BMPtr = BufBM.template get_access<sycl_read>(Cgh);

      auto KernBoolMachine = [=](sycl::id<2> Id) {
        const size_t Column = Id.get(0);
        const size_t Row = Id.get(1);
        MachineCellTy Idx = 0;
        int Shift = 0;
        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            MachineCellTy Value = 0;
            int X = Column + J;
            int Y = Row + I;
            if (X >= 0 && X < ImW && Y >= 0 && Y < ImH)
              Value = InPtr[Y][X];
            Value = Value ? 1 : 0;
            Idx += Value << Shift;
            Shift += 1;
          }
        }
        // TODO: use here accessor not host data
        OutPtr[Row][Column] = BM.get(Idx);
      };

      Cgh.parallel_for<class boolmachine_2d_buf>(Dims, KernBoolMachine);
    });

    DeviceQueue.wait(); // or explicit host accessor to Dst
    ProfInfo.push_back(Evt);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BoolMachineBuffer>(argc, argv);
}
