//-----------------------------------------------------------------------------
//
// Jigsaw puzzles
//
//-----------------------------------------------------------------------------
//
// separate module to include all logic except drawing
//
//-----------------------------------------------------------------------------

#pragma once

#include <ctime>
#include <random>
#include <vector>

#include "jigpuzzle.hpp"

namespace jigsaw {

// special puzzle with "border" color
constexpr char BLANK = 0;

// number of directions (future extensibility to hex, and so on
constexpr int NDIRS = 4;

// directions
constexpr int DIR_LEFT = 0;
constexpr int DIR_UP = 1;
constexpr int DIR_RIGHT = 2;
constexpr int DIR_DOWN = 3;

// One puzzle in lurd form. Current means possible rotation
class puzzle_t {
  char sides[NDIRS];
  char current = 0;

public:
  template <typename Integral>
  puzzle_t(std::initializer_list<Integral> ls) : current(0) {
    assert(ls.size() == NDIRS);
    std::copy_n(ls.begin(), NDIRS, &sides[0]);
  }

  puzzle_t(char l, char u, char r, char d) : puzzle_t({l, u, r, d}) {}

  int operator[](int x) {
    assert(x >= 0 &&
           "only positive directions are possible, only 0 -- 3 are intended");
    return sides[(current + x) % NDIRS];
  }
};

// field of puzzles
class field_t {
  std::vector<puzzle_t> puzzles_;
  int fx_, fy_;

public:
  field_t(int fx, int fy) : fx_(fx), fy_(fy) {}
  puzzle_t get(int x, int y) { return puzzles_[x * fy_ + y]; }
  int get_x() const noexcept { return fx_; }
  int get_y() const noexcept { return fy_; }

public:
  static field_t generate_random(int fx, int fy, int maxcolor) {
    std::mt19937 gen(std::time(0));
    std::uniform_int_distribution<> dist(1, maxcolor);
    field_t ret(fx, fy);

    for (int i = 0; i < fx; ++i)
      for (int j = 0; j < fy; ++j) {
        int left = dist(gen), up = dist(gen), right = dist(gen),
            down = dist(gen);
        if (i == 0)
          up = BLANK;
        if (j == 0)
          left = BLANK;
        if (i == fx - 1)
          down = BLANK;
        if (j == fy - 1)
          right = BLANK;
        ret.puzzles_.emplace_back(left, up, right, down);
      }

    return ret;
  }

  static field_t generate_possible(int fx, int fy, int maxcolor) {
    std::mt19937 gen(std::time(0));
    std::uniform_int_distribution<> dist(1, maxcolor);

    // start with random
    field_t ret = generate_random(fx, fy, maxcolor);

    // pass on random field: constraint every even to make it possible
    for (int i = 0; i < fx; ++i)
      for (int j = (i % 2); j < fy; j += 2) {
        int left = BLANK, up = BLANK, right = BLANK, down = BLANK;
        if (i > 0)
          up = ret.puzzles_[(i - 1) * fy + j][DIR_DOWN];
        if (i < fx - 1)
          down = ret.puzzles_[(i + 1) * fy + j][DIR_UP];
        if (j > 0)
          left = ret.puzzles_[i * fy + j - 1][DIR_RIGHT];
        if (j < fy - 1)
          right = ret.puzzles_[i * fy + j + 1][DIR_LEFT];

        ret.puzzles_[i * fy + j] = {left, up, right, down};
      }

    return ret;
  }
};

} // namespace jigsaw