// g++ cl_hist.cc @oclw2.inc

#include <iostream>

#define cimg_use_jpeg
#include "CImg.h"
#include "cl_wrapper2.h"

const char *histkernel = STRINGIFY(__kernel void histogram(
    __global uchar *data, int num_data, __global int *histogram,
    __local int *local_hist, int num_bins) {
  int i;
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int lsize = get_local_size(0);
  int gsize = get_global_size(0);

  for (i = lid; i < num_bins; i += lsize)
    local_hist[i] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = gid; i < num_data; i += gsize)
    atomic_add(&local_hist[data[i]], 1);

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = lid; i < num_bins; i += lsize)
    atomic_add(&histogram[i], local_hist[i]);
});

void ocl_hist(oclwrap2::ocl_app_t &app, int kidx, const unsigned char *imdata,
              size_t totalsz, int *hdata, int nbins) {
  int ibuf = app.add_buffer<unsigned char>(CL_MEM_READ_ONLY, imdata, totalsz);
  int hbuf = app.add_buffer<int>(CL_MEM_READ_WRITE, nbins, 0);

  app.set_kernel_buf_arg(kidx, 0, ibuf);
  app.set_kernel_int_arg(kidx, 1, totalsz);
  app.set_kernel_buf_arg(kidx, 2, hbuf);
  app.set_kernel_localbuf_arg(kidx, 3, nbins * sizeof(int));
  app.set_kernel_int_arg(kidx, 4, nbins);

  size_t wgsz = 20; // app.max_workgroup_size();
  std::cout << "Selected work group size: " << wgsz << std::endl;

  app.exec_kernel_nd(kidx, 1, &totalsz, &wgsz);
  app.read_buffer<int>(hbuf, hdata, nbins);

  app.release_mems();
}

void normal_hist(const unsigned char *imdata, size_t totalsz, int *hdata,
                 int nbins) {
  for (int i = 0; i < totalsz; ++i) {
    assert(imdata[i] < nbins);
    hdata[imdata[i]] += 1;
  }
}

const char *imname = "luperk.jpg";

void disp_image(cimg_library::CImgDisplay &draw_disp, std::vector<int> &hist,
                int hist_max, const unsigned char *cl) {
  int nbins = hist.size();
  auto ddw = draw_disp.width();
  auto ddh = draw_disp.height();
  double mult = (double)ddh / hist_max;
  double hmult = (double)ddw / nbins;
  cimg_library::CImg<unsigned char> img(ddw, ddh, 1, 3, 255);
  for (int i = 0; i < nbins; ++i) {
    int height = static_cast<int>(hist[i] * mult);
    int xstart = static_cast<int>(i * hmult);
    int xfin = static_cast<int>((i + 1) * hmult);
    img.draw_rectangle(xstart, ddh, xfin, ddh - height, cl, 1.0f, ~0U);
  }
  img.display(draw_disp);
}

int main(int argc, char **argv) {
  constexpr int nbins = 256;
  if (argc > 1)
    imname = argv[1];

  cimg_library::CImg<unsigned char> image(imname);
  int imw = image.width();
  int imh = image.height();
  size_t totalsz = imw * imh;
  std::cout << "Loaded image " << imname << ": " << imw << " x " << imh
            << std::endl;
  std::vector<int> rhist(nbins), ghist(nbins), bhist(nbins);

  oclwrap2::ocl_app_t app;
  std::cout << "Selected platform: " << app.platform_version() << std::endl;
  std::cout << "Selected device: " << app.device_name() << std::endl;

  int pidx = app.add_programm(histkernel);
  int kidx = app.extract_kernel(pidx, "histogram");

#ifdef NOOCL
  normal_hist(image.data(), totalsz, rhist.data(), nbins);
  normal_hist(image.data() + totalsz, totalsz, ghist.data(), nbins);
  normal_hist(image.data() + 2 * totalsz, totalsz, bhist.data(), nbins);
#else
  ocl_hist(app, kidx, image.data(), totalsz, rhist.data(), nbins);
  ocl_hist(app, kidx, image.data() + totalsz, totalsz, ghist.data(), nbins);
  ocl_hist(app, kidx, image.data() + 2 * totalsz, totalsz, bhist.data(), nbins);
#endif

  auto rhist_max = *std::max_element(rhist.begin(), rhist.end());
  auto ghist_max = *std::max_element(ghist.begin(), ghist.end());
  auto bhist_max = *std::max_element(bhist.begin(), bhist.end());

  const int binwidth = 2;
  const int binheight = 400;
  cimg_library::CImgDisplay main_disp(image, "Color image");
  cimg_library::CImgDisplay draw_disp_r(nbins * binwidth, binheight,
                                        "Histogramm red channel", 0),
      draw_disp_g(nbins * binwidth, binheight, "Histogramm green channel", 0),
      draw_disp_b(nbins * binwidth, binheight, "Histogramm blue channel", 0);

  const unsigned char red[] = {255, 0, 0}, green[] = {0, 255, 0},
                      blue[] = {0, 0, 255};

  while (!main_disp.is_closed() && !draw_disp_r.is_closed() &&
         !draw_disp_g.is_closed() && !draw_disp_b.is_closed()) {
    disp_image(draw_disp_r, rhist, rhist_max, red);
    disp_image(draw_disp_g, ghist, ghist_max, green);
    disp_image(draw_disp_b, bhist, bhist_max, blue);

    // Temporize event loop
    cimg_library::cimg::wait(20);
  }
}