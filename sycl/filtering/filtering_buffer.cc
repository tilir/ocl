//------------------------------------------------------------------------------
//
// Filtering with 2D convolution kernel (SYCL vs serial CPU)
// Demonstrates non-samplers, just buffers
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

#include "filtering_testers.hpp"

using ConfigTy = sycltesters::filter::Config;

// class is used for kernel name
class filter_2d_buf;

class FilterBuffer : public sycltesters::Filter {
  using sycltesters::Filter::Bundle;
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterBuffer(cl::sycl::queue &DeviceQueue, EBundleTy ExeBundle, ConfigTy Cfg)
      : sycltesters::Filter(DeviceQueue, ExeBundle), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH,
                                   drawer::Filter &Filt) override {
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    sycl::range<2> Dims(ImW, ImH);
    sycl::buffer<sycl::float4, 1> Dst(DstData, ImW * ImH);
    sycl::buffer<sycl::float4, 1> Src(SrcData, ImW * ImH);
    Src.set_final_data(nullptr);
    int FiltSize = Filt.sqrt_size();
    int DataSize = FiltSize * FiltSize;
    int HalfWidth = FiltSize / 2;
    sycl::buffer<float, 1> FiltData(Filt.data(), DataSize);
    FiltData.set_final_data(nullptr);

    // explicit accessor types
    using ImReadTy = sycl::accessor<sycl::float4, 1, sycl_read, sycl_constant>;
    using ImWriteTy = sycl::accessor<sycl::float4, 1, sycl_write, sycl_global>;
    using FiltAccTy = sycl::accessor<float, 1, sycl_read, sycl_constant>;

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      ImReadTy InPtr(Src, Cgh);
      ImWriteTy OutPtr(Dst, Cgh);
      FiltAccTy FiltPtr(FiltData, Cgh);

      auto KernFilter = [=](sycl::id<2> Id) {
        const size_t Column = Id.get(0);
        const size_t Row = Id.get(1);
        sycl::float4 Sum = {0.0f, 0.0f, 0.0f, 1.0f};
        int FiltIndex = 0;
        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            const auto X = Column + J;
            const auto Y = Row + I;
            // we have this in-bounds check for image processing
            // sampler does it behind the scenes (in hardware really)
            if (X >= 0 && X < ImW && Y >= 0 && Y < ImH) {
              sycl::float4 Pixel = InPtr[Y * ImW + X];
              Sum[0] += Pixel[0] * FiltPtr[FiltIndex];
              Sum[1] += Pixel[1] * FiltPtr[FiltIndex];
              Sum[2] += Pixel[2] * FiltPtr[FiltIndex];
            }
            FiltIndex += 1;
          }
        }
        OutPtr[Row * ImW + Column] = Sum;
      };

      Cgh.parallel_for<class filter_2d_buf>(Dims, KernFilter);
    });

    DeviceQueue.wait(); // or explicit host accessor to Dst
    ProfInfo.push_back(Evt);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycl::kernel_id kid = sycl::get_kernel_id<class filter_2d_buf>();
  sycltesters::test_sequence<FilterBuffer>(argc, argv, kid);
}
