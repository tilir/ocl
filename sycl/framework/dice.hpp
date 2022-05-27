//------------------------------------------------------------------------------
//
// Simplify work with randomness
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <random>

namespace sycltesters {

struct Dice {
  std::uniform_int_distribution<int> Uid;

  Dice(int Min, int Max) : Uid(Min, Max) {}
  int operator()() {
    static std::random_device Rd;
    static std::mt19937 Rng{Rd()};
    return Uid(Rng);
  }
};

template <typename It>
void rand_initialize(It Begin, It End, int Min, int Max) {
  Dice D(Min, Max);
  std::generate(Begin, End, [&] { return D(); });
}

} // namespace sycltesters