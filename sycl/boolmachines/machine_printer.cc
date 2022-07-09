#include <array>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <fstream>

// input: array of tuple, array of borders N0, N1, ... Nj
// modifies: tuple [first, last)
// returns: 0 if results dropped back to orig, 1 if next tuple generated
template <typename It, typename Bit>
int next_mm_tuple_of(It first, It last, Bit bfirst) {
  int j = last - first - 1;
  if (j < 0)
    return 0;

  // simplest case: just increase
  if (first[j] < bfirst[j] - 1) {
    first[j] += 1;
    return 1;
  }

  // carry case: fining j to increase
  while (j >= 0 && first[j] == bfirst[j] - 1) {
    first[j] = 0;
    j = j - 1;
  }

  // digit position found
  if (j >= 0) {
    first[j] += 1;
    return 1;
  }

  // not found: we already zeroed everything
  return 0;
}

// X[0] X[1] X[2]
// X[3] X[4] X[5]
// X[6] X[7] X[8]


// identity is X[4]
template<typename It>
unsigned char identity(It X) {
  return X[4];
}

// totalistic rule X of Wolphram
// X[1] + X[3] + X[4] + X[5] + X[7] : число 0 - 5
// say rule 38:
// 5 4 3 2 1 0
// 1 0 0 1 1 0
template<typename It>
unsigned char totalistic(It X, int N) {
  int Total = X[1] + X[3] + X[4] + X[5] + X[7];
  return (N >> Total) & 1;
}

// Conway's life is 2 < Sum(X[i]) + 0.5 * X[4] < 4
template<typename It>
unsigned char conway(It X) {
  double Sum = 0.0;
  for (int I = 0; I < 9; ++I)
    Sum += X[I];
  Sum -= 0.5 * X[4];

  if ((Sum > 2) && (Sum < 4))
    return 1;
  return 0;
}

// to switch what we are generating
auto TryMachine = [] (auto X, auto N) { return conway(X); };
const char *BMName = "conway.bm";

int main() {
  std::array<unsigned, 9> X = {0};
  std::array<unsigned, 9> Bounds;
  std::fill(Bounds.begin(), Bounds.end(), 2);

  std::ostream_iterator<unsigned> Os{std::cout, " "};

  int K = 1, Count = 0;
  unsigned char State[64] = {0};

  while (K != 0) {
    unsigned char NextBit = TryMachine(X.begin(), 9);
    State[Count / 8] |= (NextBit << (Count % 8));

#if VISUALIZE
    std::cout << (Count / 8) << ": ";
    std::copy(X.begin(), X.end(), Os);
    std::cout << " => " << static_cast<int>(State[Count / 8]);
    std::cout << std::endl;
#endif

    K = next_mm_tuple_of(X.begin(), X.end(), Bounds.begin());
    Count += 1;
  }

  std::ofstream BMStr(BMName);

  BMStr << std::hex << std::setw(2) << std::setfill('0');
  for (int I = 0; I < 64; ++I) 
    BMStr << std::setw(2) << static_cast<int>(State[I]) << " ";
  BMStr << std::endl;
}

