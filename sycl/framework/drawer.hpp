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
#include <algorithm>
#include <numeric>

// clang-format off
#define cimg_use_jpeg
#include "CImg.h"
// clang-format on

#include "dice.hpp"

using ImageTy = cimg_library::CImg<unsigned char>;

namespace drawer {

static const unsigned char red[] = {255, 0, 0};
static const unsigned char green[] = {0, 255, 0};
static const unsigned char blue[] = {0, 0, 255};

// display buffer in histogramm mode
template <typename T>
void disp_buffer(cimg_library::CImgDisplay &Display, const T *Buf, int Width,
                 int Height, const unsigned char *Cl) {
  double Ddw = Display.width();
  double Ddh = Display.height();
  double Hmult = Ddh / Height;
  double Wmult = Ddw / Width;
  ImageTy Img(Ddw, Ddh, 1, 3, 255);
  for (int I = 0; I < Width; ++I) {
    int BarHeight = Buf[I] * Hmult;
    int Xstart = I * Wmult;
    int Xfin = (I + 1) * Wmult;
    Img.draw_rectangle(Xstart, Ddh, Xfin, Ddh - BarHeight, Cl, 1.0f, ~0U);
  }
  Img.display(Display);
}

constexpr float Normalize = 255.0f;

// single-channel CImg to float array
template <typename T> inline void img_to_scalar(ImageTy &Img, T *Buf) {
  for (int Y = 0; Y < Img.height(); Y++)
    for (int X = 0; X < Img.width(); X++)
      *Buf++ = Img(X, Y, 0, 0) / Normalize;
}

// CImg to float4 array
inline void img_to_float4(ImageTy &Img, sycl::float4 *Buf) {
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
inline void float4_to_img(const sycl::float4 *Buf, ImageTy &Img) {
  for (int Y = 0; Y < Img.height(); Y++) {
    for (int X = 0; X < Img.width(); X++) {
      sycl::float4 Data = *Buf++;
      Img(X, Y, 0, 0) = clamp_uchar(Data[0] * Normalize);
      Img(X, Y, 0, 1) = clamp_uchar(Data[1] * Normalize);
      Img(X, Y, 0, 2) = clamp_uchar(Data[2] * Normalize);
    }
  }
}

template <typename T> void scalar_to_img(const T *Buf, ImageTy &Img) {
  for (int Y = 0; Y < Img.height(); Y++) {
    for (int X = 0; X < Img.width(); X++) {
      T Data = *Buf++;
      Img(X, Y, 0, 0) = clamp_uchar(Data * Normalize);
    }
  }
}

enum class ItemPatternTy {
  Filled,
  OutLined,
};

// put N random boxes, uniformly distributed
// RGB image version
inline void random_boxes_rgb(int N, ImageTy &Img, int BoxH, int BoxW,
                             ItemPatternTy Style = ItemPatternTy::Filled) {
  sycltesters::Dice DH(0, BoxH - 1), DW(0, BoxW - 1), DC(0, 255);
  for (int I = 0; I < N; ++I) {
    unsigned char color[3] = {DC(), DC(), DC()};
    int X0 = DW(), X1 = DW(), Y0 = DH(), Y1 = DH();
    int XU = std::min(X0, X1), XD = std::max(X0, X1);
    int YU = std::min(Y0, Y1), YD = std::max(Y0, Y1);
    if (Style == ItemPatternTy::Filled)
      Img.draw_rectangle(XU, YU, XD, YD, color);
    else
      Img.draw_rectangle(XU, YU, XD, YD, color, 1.0f, ~0u);
  }
}

// put N random boxes, uniformly distributed
// monotonic image version
inline void random_boxes_mono(int N, ImageTy &Img, int BoxH, int BoxW,
                              ItemPatternTy Style = ItemPatternTy::Filled) {
  sycltesters::Dice DH(0, BoxH - 1), DW(0, BoxW - 1);
  for (int I = 0; I < N; ++I) {
    unsigned char color[1] = {0};
    int X0 = DW(), X1 = DW(), Y0 = DH(), Y1 = DH();
    int XU = std::min(X0, X1), XD = std::max(X0, X1);
    int YU = std::min(Y0, Y1), YD = std::max(Y0, Y1);
    if (Style == ItemPatternTy::Filled)
      Img.draw_rectangle(XU, YU, XD, YD, color);
    else
      Img.draw_rectangle(XU, YU, XD, YD, color, 1.0f, ~0u);
  }
}

inline void random_boxes(int N, ImageTy &Img) {
  random_boxes_rgb(N, Img, Img.height(), Img.width());
}

// 2D convolution kernel
class Filter {
  int N;
  std::vector<float> Data;

public:
  Filter() : N(0) {}

  // random filter
  Filter(int RandSz, int MinFilt, int MaxFilt) : N(RandSz), Data(N * N) {
    sycltesters::rand_initialize(Data.begin(), Data.end(), MinFilt, MaxFilt);
    float NormVal = std::reduce(Data.begin(), Data.end());
    if (abs(NormVal) > 1.0)
      std::transform(Data.begin(), Data.end(), Data.begin(),
                     [NormVal](auto Elt) { return Elt / NormVal; });
  }

  // load from file
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
