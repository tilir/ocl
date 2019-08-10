#include <iostream>
#include <vector>

#define cimg_use_jpeg
#include "CImg.h"
#include "cl_wrapper2.h"

const char *rotkernel = STRINGIFY(
    __kernel void rotation(__global uchar *input, __global uchar *output,
                           int width, int height, float theta) {
      int x, y, sz;
      int xr, yr;
      float x0, y0, xprime, yprime;

      x = get_global_id(0);
      y = get_global_id(1);
      sz = width * height;

      x0 = width / 2.0f;
      y0 = height / 2.0f;

      xprime = x - x0;
      yprime = y - y0;

      xr = xprime * cos(theta) - yprime * sin(theta) + x0;
      yr = xprime * sin(theta) + yprime * cos(theta) + y0;

      output[y * width + x + sz * 0] = 0;
      output[y * width + x + sz * 1] = 0;
      output[y * width + x + sz * 2] = 0;
      if ((xr < width) && (yr < height) && (xr > 0) && (yr > 0)) {
        int npos = yr * width + xr;
        output[y * width + x + sz * 0] = input[npos + sz * 0];
        output[y * width + x + sz * 1] = input[npos + sz * 1];
        output[y * width + x + sz * 2] = input[npos + sz * 2];
      }
    });

const char *imname = "luperk.jpg";

cimg_library::CImg<unsigned char>
ocl_rotate(oclwrap2::ocl_app_t &app, int kidx,
           cimg_library::CImg<unsigned char> image, float theta) {
  int imw = image.width();
  int imh = image.height();
  int imd = image.depth();
  int ims = image.spectrum();
  size_t imsz = imw * imh * ims;

  cimg_library::CImg<unsigned char> outimage(imw, imh, imd, ims);

  int inimg =
      app.add_buffer<unsigned char>(CL_MEM_READ_ONLY, image.data(), imsz);
  int outimg = app.add_buffer<unsigned char>(CL_MEM_WRITE_ONLY, NULL, imsz);

  app.set_kernel_buf_arg(kidx, 0, inimg);
  app.set_kernel_buf_arg(kidx, 1, outimg);
  app.set_kernel_int_arg(kidx, 2, imw);
  app.set_kernel_int_arg(kidx, 3, imh);
  app.set_kernel_float_arg(kidx, 4, theta);

  size_t globalws[2] = {static_cast<size_t>(imw), static_cast<size_t>(imh)};
  size_t localws[2] = {1, 1};

  app.exec_kernel_nd(kidx, 2, globalws, localws);
  app.read_buffer<unsigned char>(outimg, outimage.data(), imsz);

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

  const float theta2 = 5.0f * 3.141592f / 180.0f;
  auto interpimage = ocl_rotate(app, kidx, image, theta2);
  for (int i = 1; i < 72; ++i)
    interpimage = ocl_rotate(app, kidx, interpimage, theta2);

  cimg_library::CImgDisplay main_disp(image, "Input image");
  cimg_library::CImgDisplay out_disp(outimage, "Resulting image");
  cimg_library::CImgDisplay iout_disp(interpimage, "Interp image");

  while (!main_disp.is_closed()) {
    cimg_library::cimg::wait(20);
  }
}
