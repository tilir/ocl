//------------------------------------------------------------------------------
//
// Bitonic sort, SYCL way, with explicit buffers
// no explicit sync required
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "bitonic_testers.hpp"

template <typename T>
class BitonicSortBuf : public sycltesters::BitonicSort<T> {
  using sycltesters::BitonicSort<T>::Queue;

public:
  BitonicSortBuf(cl::sycl::queue &DeviceQueue)
      : sycltesters::BitonicSort<T>(DeviceQueue) {}

  sycltesters::EvtRet_t operator()(T *Vec, size_t Sz) override { return {}; }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BitonicSortBuf<int>>(argc, argv);
}
