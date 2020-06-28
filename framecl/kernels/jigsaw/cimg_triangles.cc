//-----------------------------------------------------------------------------
//
// triangles in cimg
//
//-----------------------------------------------------------------------------
//
// jigsaw puzzle visualization
// triangles in all positions with all pictures: opacity and other problems
//
//-----------------------------------------------------------------------------

#include <iostream>
#include <stdexcept>

// clang-format off
#define cimg_use_png
#include "CImg.h"
// clang-format on

#include "jigdraw.hpp"

const int BLOCK_WIDTH = 256;
const int BLOCK_HEIGHT = 128;
const int BLOCKS_X = 5;
const int BLOCKS_Y = 5;

// TextureList ctor resizes by 1/2
const int RESIZED_WIDTH = 128;
const int RESIZED_HEIGHT = 64;

int triamain(int argc, char **argv) {
  int width = BLOCK_WIDTH * BLOCKS_X;
  int height = BLOCK_HEIGHT * BLOCKS_Y;

  cimg_library::CImgDisplay main_disp(width, height, "Triangles");
  jigsaw::TextureList pieces("pieces.png", BLOCK_WIDTH, BLOCK_HEIGHT);

  while (!main_disp.is_closed()) {
    cimg_library::CImg<unsigned char> img(main_disp.width(), main_disp.height(),
                                          /* size z */ 1, /* size c */ 3,
                                          /* fill */ 255);

    // pieces #(0 -- 23) with opacity
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j) {
        int k = i * 3 + j;
        pieces.draw(4 * k, 0, img, RESIZED_WIDTH * j, 2 * RESIZED_HEIGHT * i);
        pieces.draw(4 * k + 1, 90, img, RESIZED_WIDTH * j + RESIZED_WIDTH / 2,
                    2 * RESIZED_HEIGHT * i);
        pieces.draw(4 * k + 2, 180, img, RESIZED_WIDTH * j,
                    2 * RESIZED_HEIGHT * i + RESIZED_HEIGHT);
        pieces.draw(4 * k + 3, 270, img, RESIZED_WIDTH * j,
                    2 * RESIZED_HEIGHT * i);
      }

    img.display(main_disp);
    cimg_library::cimg::wait(20);
  }

  return 0;
}

int main(int argc, char **argv) {
  try {
    return triamain(argc, argv);
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
  return -1;
}
