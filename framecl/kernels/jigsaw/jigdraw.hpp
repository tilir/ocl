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

#include <array>
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
  int piecex_, piecey_;

public:
  TextureList(const char *fname, int nx, int ny) : texture_(fname) {
    int w = texture_.width();
    int h = texture_.height();
    assert(((w % nx) == 0) && "width shall be equally divided by nx pieces");
    assert(((h % ny) == 0) && "height shall be equally divided by ny pieces");

    piecex_ = w / nx;
    piecey_ = h / ny;

    std::cout << "Loaded texture: " << w << " x " << h << std::endl;
    std::cout << "Channels: " << texture_.spectrum() << std::endl;
    for (int x = 0; x < nx; ++x)
      for (int y = 0; y < ny; ++y)
        splitted_.push_back(texture_
                                .get_crop(x * piecex_, y * piecey_,
                                          (x + 1) * piecex_ - 1,
                                          (y + 1) * piecey_ - 1)
                                .resize_halfXY());

    // because we did resize
    piecex_ /= 2;
    piecey_ /= 2;
  }

  int get_x() const noexcept { return piecex_; }
  int get_y() const noexcept { return piecey_; }

  // draw image for triangle with rotation
  void draw(char num, int angle, cimg_library::CImg<unsigned char> &img, int x,
            int y) {
    assert((angle % 90) == 0 && "only right angle rotations available");
    cimg_library::CImg<unsigned char> selected = splitted_.at(num);
    selected.rotate(angle, selected.width() / 2, selected.height() / 2);
    img.draw_image(x, y, selected.get_shared_channels(0, 2),
                   selected.get_shared_channel(3), 1, 255U);
  }

  // draw square of 4 triangles: lurd
  void drawsq(puzzle_t nums, cimg_library::CImg<unsigned char> &img, int x,
              int y) {
    draw(nums[DIR_LEFT], 270, img, x, y);
    draw(nums[DIR_UP], 0, img, x, y);
    draw(nums[DIR_RIGHT], 90, img, x + piecex_ / 2, y);
    draw(nums[DIR_DOWN], 180, img, x, y + piecey_);
  }

  void draw_field(field_t f, cimg_library::CImg<unsigned char> &img, int sqx,
                  int sqy) {
    for (int i = 0; i < f.get_x(); ++i)
      for (int j = 0; j < f.get_y(); ++j)
        drawsq(f.get(i, j), img, sqx * j, sqy * i);
  }
};

} // namespace jigsaw