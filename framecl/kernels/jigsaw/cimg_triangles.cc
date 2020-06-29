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

#include <ctime>
#include <iostream>
#include <random>
#include <stdexcept>

// clang-format off
#define cimg_use_png
#include "CImg.h"
// clang-format on

#include "jigdraw.hpp"

constexpr const char *texturename = "pieces.png";
constexpr const int pieces_x = 4;
constexpr const int pieces_y = 6;

constexpr int field_x = 8;
constexpr int field_y = 8;

constexpr int max_piece = 22;

int triamain(int argc, char **argv) {
  jigsaw::TextureList pieces(texturename, pieces_x, pieces_y);

  int rw = pieces.get_x();
  int rh = pieces.get_y();

  int sqx = rw;
  int sqy = rh * 2;

  cimg_library::CImgDisplay main_disp(sqx * field_x, sqy * field_y,
                                      "Triangles");

  jigsaw::field_t fld =
      jigsaw::field_t::generate_possible(field_x, field_y, max_piece);

  while (!main_disp.is_closed()) {
    cimg_library::CImg<unsigned char> img(main_disp.width(), main_disp.height(),
                                          /* size z */ 1, /* size c */ 3,
                                          /* fill */ 255);

    pieces.draw_field(fld, img, sqx, sqy);
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
