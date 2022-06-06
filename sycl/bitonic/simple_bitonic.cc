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

template <typename T> void bitonic_sort(T *A, size_t Sz) {
  assert(A);
  int Step, Stage;

  if (std::popcount(Sz) != 1 || Sz < 2)
    throw std::runtime_error("Please use only power-of-two arrays");

  int N = std::countr_zero(Sz);

  for (Step = 0; Step < N; Step++) {
    std::cout << "Step " << Step << ": " << std::endl;
    for (Stage = Step; Stage >= 0; Stage--) {
      const int SeqLen = 1 << (Stage + 1);
      const int Power2 = 1 << (Step - Stage);
      std::cout << "Stage: " << Stage << " : " << SeqLen << " : " << Power2
                << std::endl;

      for (int I = 0; I < Sz; ++I) {
        const int SeqNum = I / SeqLen;
        const int Odd = SeqNum / Power2;
        const bool Increasing = ((Odd % 2) == 0);
        const int HalfLen = SeqLen / 2;

        if (I < (SeqLen * SeqNum) + HalfLen) {
          const int J = I + HalfLen;
          std::cout << "(" << I << ", " << J
                    << ") : " << (SeqLen * SeqNum) + HalfLen << " ";
          if (((A[I] > A[J]) && Increasing) || ((A[I] < A[J]) && !Increasing)) {
            T Temp = A[I];
            A[I] = A[J];
            A[J] = Temp;
          }
        }
      }
      std::cout << std::endl;
    }
    std::cout << "After step " << Step << ": " << std::endl;
    visualize_seq(A, A + Sz, std::cout);
  }
}

// test for SZ-element arrays with visualization
constexpr int SZ = 32;

int main() {
  std::vector<int> v = {20, 22, 2,  19, 1,  16, 9, 0,  12, 24, 18,
                        8,  16, 4,  24, 29, 4,  5, 24, 0,  15, 20,
                        16, 9,  15, 2,  17, 32, 8, 11, 28, 19};

  // std::vector<int> v(SZ);
  // Dice d{0, SZ};
  // std::generate(v.begin(), v.end(), [&] { return d(); });
  std::cout << "Initial: " << std::endl;
  visualize_seq(v.begin(), v.end(), std::cout);
  bitonic_sort(v.data(), v.size());
  std::cout << "Final: " << std::endl;
  visualize_seq(v.begin(), v.end(), std::cout);
}