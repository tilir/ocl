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

#include "jigpuzzle.hpp"

namespace jigsaw {

constexpr char BLANKNUM = 35;

struct puzzle_t {
  char sides[4];
  char current = 0;
};

} // namespace jigsaw