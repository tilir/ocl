//------------------------------------------------------------------------------
//
// Generic code to draw neat pictures with CImg
// Avoiding tons of boilerplate otherwise
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

// clang-format off
#define cimg_use_jpeg
#include "CImg.h"
// clang-format on

namespace drawer {

static const unsigned char red[] = {255, 0, 0};
static const unsigned char green[] = {0, 255, 0};
static const unsigned char blue[] = {0, 0, 255};

template <typename T>
void disp_buffer(cimg_library::CImgDisplay &draw_disp, T *buf, int bufw,
                 int bufh, const unsigned char *cl) {
  double ddw = draw_disp.width();
  double ddh = draw_disp.height();
  double hmult = ddh / bufh;
  double wmult = ddw / bufw;
  cimg_library::CImg<unsigned char> img(ddw, ddh, 1, 3, 255);
  for (int i = 0; i < bufw; ++i) {
    int height = buf[i] * hmult;
    int xstart = i * wmult;
    int xfin = (i + 1) * wmult;
    img.draw_rectangle(xstart, ddh, xfin, ddh - height, cl, 1.0f, ~0U);
  }
  img.display(draw_disp);
}

} // namespace drawer
