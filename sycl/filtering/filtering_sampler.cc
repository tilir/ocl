//------------------------------------------------------------------------------
//
// Filtering with 2D convolution kernel (SYCL vs serial CPU)
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

#include "filtering_testers.hpp"

using ConfigTy = sycltesters::filter::Config;

// class is used for kernel name
class filter_2d_vec;

class FilterSamplerVec : public sycltesters::Filter {
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterSamplerVec(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Filter(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH,
                                   drawer::Filter &Filt) override {
    sycltesters::EvtVec_t ProfInfo;
    sycl::range<2> Dims(ImW, ImH);
    auto &DeviceQueue = Queue();
    sycl::image<2> Dst(DstData, sycl_rgba, sycl_fp32, Dims);
    sycl::image<2> Src(SrcData, sycl_rgba, sycl_fp32, Dims);
    int FiltSize = Filt.sqrt_size();
    int DataSize = FiltSize * FiltSize;
    int HalfWidth = FiltSize / 2;

    sycl::buffer<sycl::float4, 1> FiltBuffer(DataSize);
    {
      auto FiltHostAcc = FiltBuffer.template get_access<sycl_write>();
      const float *FiltData = Filt.data();
      for (int I = 0; I < DataSize; ++I) {
        float FiltChannel = FiltData[I];
        FiltHostAcc[I] =
            sycl::float4{FiltChannel, FiltChannel, FiltChannel, 1.0f};
      }
    }

    using ImAccTy = sycl::accessor<sycl::float4, 2, sycl_read, sycl_image>;
    using ImWriteTy = sycl::accessor<sycl::float4, 2, sycl_write, sycl_image>;

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      ImAccTy InPtr(Src, Cgh);
      ImWriteTy OutPtr(Dst, Cgh);
      auto FiltPtr = FiltBuffer.template get_access<sycl_read>(Cgh);

      sycl::sampler Sampler(sycl::coordinate_normalization_mode::unnormalized,
                            sycl::addressing_mode::clamp,
                            sycl::filtering_mode::nearest);

      auto KernFilter = [=](sycl::id<2> Id) {
        const size_t Column = Id.get(0);
        const size_t Row = Id.get(1);
        sycl::float4 Sum = {0.0f, 0.0f, 0.0f, 1.0f};
        int FiltIndex = 0;
        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            // first changed index is X
            sycl::int2 Coords = {Column + J, Row + I};
            sycl::float4 Pixel = InPtr.read(Coords, Sampler);

            // vectorized multiplication
            Sum += Pixel * FiltPtr[FiltIndex];
            FiltIndex += 1;
          }
        }
        sycl::int2 Coords = {Column, Row};
        OutPtr.write(Coords, Sum);
      };

      Cgh.parallel_for<class filter_2d_vec>(Dims, KernFilter);
    });

    ProfInfo.push_back(Evt);
    DeviceQueue.wait(); // or need host acessor to image here
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<FilterSamplerVec>(argc, argv);
}
