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

#include <CL/sycl.hpp>

// clang-format off
#define cimg_use_jpeg
#include "CImg.h"
// clang-format on

namespace drawer {

static const unsigned char red[] = {255, 0, 0};
static const unsigned char green[] = {0, 255, 0};
static const unsigned char blue[] = {0, 0, 255};

// display buffer in histogramm mode
template <typename T>
void disp_buffer(cimg_library::CImgDisplay &Display, T *Buf, int Width,
                 int Height, const unsigned char *Cl) {
  double Ddw = Display.width();
  double Ddh = Display.height();
  double Hmult = Ddh / Height;
  double Wmult = Ddw / Width;
  cimg_library::CImg<unsigned char> Img(Ddw, Ddh, 1, 3, 255);
  for (int I = 0; I < Width; ++I) {
    int BarHeight = Buf[I] * Hmult;
    int Xstart = I * Wmult;
    int Xfin = (I + 1) * Wmult;
    Img.draw_rectangle(Xstart, Ddh, Xfin, Ddh - BarHeight, Cl, 1.0f, ~0U);
  }
  Img.display(Display);
}

constexpr float Normalize = 255.0f;

// CImg to float4 array
inline void img_to_float4(cimg_library::CImg<unsigned char> &Img,
                          sycl::float4 *Buf) {
  for (int Y = 0; Y < Img.height(); Y++)
    for (int X = 0; X < Img.width(); X++) {
      sycl::float4 Data(Img(X, Y, 0, 0) / Normalize,
                        Img(X, Y, 0, 1) / Normalize,
                        Img(X, Y, 0, 2) / Normalize, 0.0f);
      *Buf++ = Data;
    }
}

inline unsigned char clamp_uchar(float f) {
  if (f <= 0.0)
    return 0;
  if (f >= 255.0)
    return 255;
  return static_cast<unsigned char>(f);
}

// float4 array to CImg
inline void float4_to_img(sycl::float4 *Buf,
                          cimg_library::CImg<unsigned char> &Img) {
  for (int Y = 0; Y < Img.height(); Y++)
    for (int X = 0; X < Img.width(); X++) {
      sycl::float4 Data = *Buf++;
      Img(X, Y, 0, 0) = clamp_uchar(Data[0] * Normalize);
      Img(X, Y, 0, 1) = clamp_uchar(Data[1] * Normalize);
      Img(X, Y, 0, 2) = clamp_uchar(Data[2] * Normalize);
    }
}

// 2D convolution kernel
class Filter {
  int N;
  std::vector<float> Data;

public:
  Filter(std::string FilterPath) {
    std::ifstream Is(FilterPath);
    Is.exceptions(std::istream::failbit);
    float NormVal;
    Is >> N;
    Is >> NormVal;
    Data.resize(N * N);
    for (int I = 0; I < N * N; ++I) {
      float Val;
      Is >> Val;
      Data[I] = Val / NormVal;
    }
  }
  int sqrt_size() const { return N; }
  const float *data() const { return Data.data(); }
};

} // namespace drawer
