//------------------------------------------------------------------------------
//
// Useful program to illustrate how bitonic sort works.
// For decent array, fitting into screen, it does all steps.
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

template <typename It, typename Os>
void visualize_seq(It begin, It end, Os &stream) {
  using T = typename std::iterator_traits<It>::value_type;
  std::ostream_iterator<T> Out{stream, " "};
  std::copy(begin, end, Out);
  stream << "\n";
}

struct Dice {
  std::uniform_int_distribution<int> uid;

  Dice(int min, int max) : uid(min, max) {}
  int operator()() {
    static std::random_device rd;
    static std::mt19937 rng{rd()};
    return uid(rng);
  }
};

template <typename T>
void SwapElements(T *Vec, int NSeq, int SeqLen, int Power2) {
  for (int SNum = 0; SNum < NSeq; SNum++) {
    int Odd = SNum / Power2;
    bool Increasing = ((Odd % 2) == 0);
    int HalfLen = SeqLen / 2;

    // For all elements in a bitonic sequence, swap them if needed
    for (int I = SNum * SeqLen; I < SNum * SeqLen + HalfLen; I++) {
      int J = I + HalfLen;
      std::cout << "(" << I << ", " << J << ") ";
      if (((Vec[I] > Vec[J]) && Increasing) ||
          ((Vec[I] < Vec[J]) && !Increasing))
        std::swap(Vec[I], Vec[J]);
    }
  }
  std::cout << std::endl;
}

template <typename T> void bitonic_sort(T *Vec, size_t Sz) {
  assert(Vec);
  int NSeq, SeqLen, Step, Stage, Power2;

  if (std::popcount(Sz) != 1 || Sz < 2)
    throw std::runtime_error("Please use only power-of-two arrays");

  int N = std::countr_zero(Sz);

  for (Step = 0; Step < N; Step++) {
    std::cout << "Step " << Step << ": " << std::endl;
    for (Stage = Step; Stage >= 0; Stage--) {
      std::cout << "Stage: " << Stage << ": " << std::endl;
      NSeq = 1 << (N - Stage - 1);
      SeqLen = 1 << (Stage + 1);
      Power2 = 1 << (Step - Stage);
      SwapElements(Vec, NSeq, SeqLen, Power2);
    }
    std::cout << "After step " << Step << ": " << std::endl;
    visualize_seq(Vec, Vec + Sz, std::cout);
  }
}

// test for SZ-element arrays with visualization
constexpr int SZ = 32;

int main() {
  std::vector<int> v(SZ);
  Dice d{0, SZ};
  std::generate(v.begin(), v.end(), [&] { return d(); });
  std::cout << "Initial: " << std::endl;
  visualize_seq(v.begin(), v.end(), std::cout);
  bitonic_sort(v.data(), v.size());
  std::cout << "Final: " << std::endl;
  visualize_seq(v.begin(), v.end(), std::cout);
}