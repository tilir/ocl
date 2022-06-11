//------------------------------------------------------------------------------
//
// Rotateing with 2D convolution kernel (SYCL vs serial CPU)
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

#include "rotate_testers.hpp"

using ConfigTy = sycltesters::rotate::Config;

// class is used for kernel name
class rotate_2d_sampler;

class RotateSamplerVec : public sycltesters::Rotate {
  using sycltesters::Rotate::Queue;
  ConfigTy Cfg_;

public:
  RotateSamplerVec(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Rotate(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH, float Theta) override {
    sycltesters::EvtVec_t ProfInfo;
    std::copy(SrcData, SrcData + ImW * ImH, DstData);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<RotateSamplerVec>(argc, argv);
}
