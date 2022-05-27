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
class filter_2d;

class FilterSampler : public sycltesters::Filter {
  using sycltesters::Filter::Bundle;
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterSampler(cl::sycl::queue &DeviceQueue, EBundleTy ExeBundle, ConfigTy Cfg)
      : sycltesters::Filter(DeviceQueue, ExeBundle), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH,
                                   drawer::Filter &Filt) override {
    sycltesters::EvtVec_t ProfInfo;
    sycl::range<2> Dims(ImW, ImH);
    auto &DeviceQueue = Queue();
    sycl::image<2> Dst(DstData, sycltesters::sycl_rgba, sycltesters::sycl_fp32,
                       Dims);
    sycl::image<2> Src(SrcData, sycltesters::sycl_rgba, sycltesters::sycl_fp32,
                       Dims);
    int FiltSize = Filt.sqrt_size();
    int HalfWidth = FiltSize / 2;
    cl::sycl::buffer<float, 1> FiltData(Filt.data(), FiltSize * FiltSize,
                                        {host_ptr});

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      using ImAccTy = sycl::accessor<sycl::float4, 2, sycl_read, sycl_image>;
      ImAccTy InPtr(Src, Cgh);
      auto OutPtr = Dst.template get_access<sycl::float4, sycl_write>(Cgh);
      auto FiltPtr = FiltData.template get_access<sycl_read>(Cgh);
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
            // int FiltIndex = (I + HalfWidth) * FiltSize + (J + HalfWidth);
            // first changed index is X
            sycl::int2 Coords = {Column + J, Row + I};
            sycl::float4 Pixel = InPtr.read(Coords, Sampler);
            Sum[0] += Pixel[0] * FiltPtr[FiltIndex];
            Sum[1] += Pixel[1] * FiltPtr[FiltIndex];
            Sum[2] += Pixel[2] * FiltPtr[FiltIndex];
            FiltIndex += 1;
          }
        }
        sycl::int2 Coords = {Column, Row};
        OutPtr.write(Coords, Sum);
      };

      Cgh.parallel_for<class filter_2d>(Dims, KernFilter);
    });

    ProfInfo.push_back(Evt);

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
#if 0
  auto ids = sycl::get_kernel_ids();
  sycl::kernel_id kid = ids[0];
  std::cout << "Registered kernels:\n";
  for (auto elt : ids)
    std::cout << elt.get_name() << std::endl;
#endif
  sycl::kernel_id kid = sycl::get_kernel_id<class filter_2d>();

  sycltesters::test_sequence<FilterSampler>(argc, argv, kid);
}
