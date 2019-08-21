#include <iostream>
#include <vector>

#define cimg_use_jpeg
#include "CImg.h"
#include "cl_wrapper2.h"

const char *rotkernel = STRINGIFY(
    __constant sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;

    __kernel void rotation(__read_only image2d_t input_image,
                           __write_only image2d_t output_image, int width,
                           int height, float theta) {
      int x, y, xprime, yprime;
      float x0, y0;
      int2 out_coords;
      float2 read_coords;
      float4 value;

      x = get_global_id(0);
      y = get_global_id(1);

      x0 = width / 2.0f;
      y0 = height / 2.0f;

      xprime = x - x0;
      yprime = y - y0;

      read_coords.x = xprime * cos(theta) - yprime * sin(theta) + x0;
      read_coords.y = xprime * sin(theta) + yprime * cos(theta) + y0;
      value = read_imagef(input_image, sampler, read_coords);

      out_coords.x = x;
      out_coords.y = y;
      write_imagef(output_image, out_coords, value);
    });

const char *imname = "luperk.jpg";

cimg_library::CImg<unsigned char>
ocl_rotate(oclwrap2::ocl_app_t &app, int kidx,
           cimg_library::CImg<unsigned char> image, float theta) {
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

  app.set_kernel_buf_arg(kidx, 0, inimg);
  app.set_kernel_buf_arg(kidx, 1, outimg);
  app.set_kernel_int_arg(kidx, 2, imw);
  app.set_kernel_int_arg(kidx, 3, imh);
  app.set_kernel_float_arg(kidx, 4, theta);

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

  int pidx = app.add_programm(rotkernel);
  int kidx = app.extract_kernel(pidx, "rotation");

  cimg_library::CImg<unsigned char> image(imname);
  std::cout << "Loaded image " << imname << std::endl;

  const float theta = 45.0f * 3.141592f / 180.0f;
  auto outimage = ocl_rotate(app, kidx, image, theta);

  const float theta2 = 1.0f * 3.141592f / 180.0f;
  auto interpimage = ocl_rotate(app, kidx, image, theta2);
  for (int i = 1; i < 45; ++i)
    interpimage = ocl_rotate(app, kidx, interpimage, theta2);

  cimg_library::CImgDisplay main_disp(image, "Input image");
  cimg_library::CImgDisplay out_disp(outimage, "Resulting image");
  cimg_library::CImgDisplay iout_disp(interpimage, "Interp image");

  while (!main_disp.is_closed()) {
    cimg_library::cimg::wait(20);
  }
}
