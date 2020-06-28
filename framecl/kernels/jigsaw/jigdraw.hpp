//-----------------------------------------------------------------------------
//
// Jigsaw puzzles drawing logic
//
//-----------------------------------------------------------------------------
//
// note cimg_use_png and required png libray in project settings
//
//-----------------------------------------------------------------------------

#pragma once

#include <cassert>

// clang-format off
#define cimg_use_png
#include "CImg.h"
// clang-format on

#include "jigpuzzle.hpp"

namespace jigsaw {

class TextureList {
  cimg_library::CImg<unsigned char> texture_;
  cimg_library::CImgList<unsigned char> splitted_;

public:
  TextureList(const char *fname, int stepx, int stepy) : texture_(fname) {
    int w = texture_.width();
    int h = texture_.height();
    std::cout << "Loaded texture: " << w << " x " << h << std::endl;
    std::cout << "Channels: " << texture_.spectrum() << std::endl;
    for (int x = 0; x < w; x += stepx)
      for (int y = 0; y < h; y += stepy)
        splitted_.push_back(
            texture_.get_crop(x, y, x + stepx - 1, y + stepy - 1)
                .resize_halfXY());
  }

  // draw image for triangle with rotation
  void draw(char num, int angle, cimg_library::CImg<unsigned char> &img, int x,
            int y) {
    assert((angle % 90) == 0 && "only right angle rotations available");
    cimg_library::CImg<unsigned char> selected = splitted_.at(num);
    selected.rotate(angle, selected.width() / 2, selected.height() / 2);
    img.draw_image(x, y, selected.get_shared_channels(0, 2),
                   selected.get_shared_channel(3), 1, 255U);
  }
};

} // namespace jigsaw