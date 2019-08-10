#include <iostream>
#include <vector>

#define cimg_use_jpeg
#include "CImg.h"
#include "cl_wrapper2.h"

const char *convkernel = STRINGIFY(__kernel void convolution(
    __read_only image2d_t input_image, __write_only image2d_t output_image,
    int rows, int cols, __constant float *filter, int filter_width,
    sampler_t sampler) {
  int column, row, half_width, filter_idx, i, j;
  int2 coords;

  // RGBA sum
  float4 sum = {0.0f, 0.0f, 0.0f, 1.0f};

  column = get_global_id(0);
  row = get_global_id(1);
  half_width = filter_width / 2;
  filter_idx = 0;

  for (i = -half_width; i <= half_width; ++i) {
    coords.y = row + i;
    for (j = -half_width; j <= half_width; ++j) {
      float4 pixel;
      coords.x = column + j;
      pixel = read_imagef(input_image, sampler, coords);
      sum.x += pixel.x * filter[filter_idx];
      sum.y += pixel.y * filter[filter_idx];
      sum.z += pixel.z * filter[filter_idx];
      filter_idx += 1;
    }
  }

  coords.x = column;
  coords.y = row;
  write_imagef(output_image, coords, sum);
});

const char *imname = "luperk.jpg";

static float gaussianBlurFilter[25] = {
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,
    1.0f / 273.0f,  4.0f / 273.0f,  16.0f / 273.0f, 26.0f / 273.0f,
    16.0f / 273.0f, 4.0f / 273.0f,  7.0f / 273.0f,  26.0f / 273.0f,
    41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,  4.0f / 273.0f,
    16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,
    1.0f / 273.0f};

static const int gaussianBlurFilterWidth = 5;

static float sharpenFilter[9] = {-1.0f / 1.1f, -2.0f / 1.1f, -1.0f / 1.1f,
                                 0.0f,         0.0f,         0.0f,
                                 1.0f / 1.1f,  2.0f / 1.1f,  1.0f / 1.1f};

static const int sharpenFilterWidth = 3;

cimg_library::CImg<unsigned char>
ocl_filter(oclwrap2::ocl_app_t &app, int kidx,
           cimg_library::CImg<unsigned char> image, float *filter,
           int filter_width) {
  int imw = image.width();
  int imh = image.height();
  int imd = image.depth();
  int ims = image.spectrum();
  size_t imsz = imw * imh;

  cimg_library::CImg<unsigned char> outimage(imw, imh, imd, ims);

  // RGBA images to multiplex
  cimg_library::CImg<float> inp(imw * (ims + 1), imh, 1, 1),
      outp(imw * (ims + 1), imh, 1, 1);

  // multiplex image
  for (int x = 0; x < imw; ++x)
    for (int y = 0; y < imh; ++y) {
      for (int k = 0; k < ims; ++k)
        inp(x * (ims + 1) + k, y, 0, 0) = image(x, y, 0, k);
      inp(x * (ims + 1) + ims, y, 0, 0) = 1.0f;
    }

  int inimg = app.add_2d_image<float>(imw, imh, ims, inp.data(), CL_RGBA,
                                      CL_FLOAT, CL_MEM_READ_ONLY);

  int outimg = app.add_2d_image<float>(imw, imh, ims, NULL, CL_RGBA, CL_FLOAT,
                                       CL_MEM_WRITE_ONLY);

  int filterbuf = app.add_buffer<float>(CL_MEM_READ_ONLY, filter,
                                        filter_width * filter_width);

  int samplerid =
      app.add_sampler(CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);

  app.set_kernel_buf_arg(kidx, 0, inimg);
  app.set_kernel_buf_arg(kidx, 1, outimg);
  app.set_kernel_int_arg(kidx, 2, imw);
  app.set_kernel_int_arg(kidx, 3, imh);
  app.set_kernel_buf_arg(kidx, 4, filterbuf);
  app.set_kernel_int_arg(kidx, 5, filter_width);
  app.set_kernel_sampler_arg(kidx, 6, samplerid);

  size_t globalws[2] = {static_cast<size_t>(imw), static_cast<size_t>(imh)};
  size_t localws[2] = {1, 1};

  app.exec_kernel_nd(kidx, 2, globalws, localws);

  app.read_2d_image(outimg, outp.data(), imw, imh);

  // demultiplex image
  for (int x = 0; x < imw; ++x)
    for (int y = 0; y < imh; ++y)
      for (int k = 0; k < ims; ++k)
        outimage(x, y, 0, k) = outp(x * (ims + 1) + k, y, 0, 0);

  app.release_mems();

  return outimage;
}

int main(int argc, char **argv) {
  constexpr int nbins = 256;
  if (argc > 1)
    imname = argv[1];

  oclwrap2::ocl_app_t app;
  std::cout << "Selected platform: " << app.platform_version() << std::endl;
  std::cout << "Selected device: " << app.device_name() << std::endl;

  int pidx = app.add_programm(convkernel);
  int kidx = app.extract_kernel(pidx, "convolution");

  cimg_library::CImg<unsigned char> image(imname);
  std::cout << "Loaded image " << imname << std::endl;

  int filterWidth = sharpenFilterWidth; /*  gaussianBlurFilterWidth; */
  float *filter = sharpenFilter;        /* gaussianBlurFilter; */

  auto filtim = ocl_filter(app, kidx, image, filter, filterWidth);

  cimg_library::CImgDisplay main_disp(image, "Input image");
  cimg_library::CImgDisplay filt_disp(filtim, "Filtered image");

  while (!main_disp.is_closed()) {
    cimg_library::cimg::wait(20);
  }
}
