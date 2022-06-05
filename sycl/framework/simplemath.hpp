//------------------------------------------------------------------------------
//
// Some simple math routines
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

namespace simplemath {

template <typename T> T roundup(T n, T m) {
  if ((n % m) == 0)
    return n;
  return ((n / m) + 1) * m;
}

} // namespace simplemath