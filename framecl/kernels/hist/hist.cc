#include <iostream>
#include <string>

#include "framecl.hpp"

// clang-format off
#define cimg_use_jpeg
#include "CImg.h"
// clang-format on

constexpr int DEFBINS = 256;
constexpr const char *DEFIM = "luperk.jpg";

void disp_image(cimg_library::CImgDisplay &draw_disp,
                framecl::buffer_t<int> &hist, int nbins, int hist_max,
                const unsigned char *cl) {
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

int histmain(int argc, char **argv) {
  framecl::optparser_t opts;

  opts.add<int>("nbins", DEFBINS, "number of bins in histogram");
  opts.add<std::string>("image", DEFIM, "image of which to build histogram");

  opts.parse(argc, argv);

  opts.require_platform("This program requires platform specification. Use "
                        "--list for available platforms");
  opts.require_program("This program needs external cl program file. It shall "
                       "contain 'histogram' kernel");

  int nbins = opts.check<int>(
      "nbins", [](int nbins) { return nbins > 0; },
      "number of bins shall be > 0");
  std::string imname = opts.get<std::string>("image");

  if (!opts.quiet()) {
    std::cout << "Hello from hist with nbins = " << nbins << std::endl;
    std::cout << "Use --nbins option to change number of bins" << std::endl;
    std::cout << "Use --image option to customize image" << std::endl;
  }

  cimg_library::CImg<unsigned char> image(imname.c_str());
  int imw = image.width();
  int imh = image.height();
  int totalsz = imw * imh;
  int localsz = 20; // TODO: ?

  if (!opts.quiet()) {
    std::cout << "Loaded image " << imname << ": " << imw << " x " << imh
              << std::endl;
  }

  framecl::context_t ctx(opts);
  framecl::program_t prog(ctx, opts);

  cl::NDRange offset{cl::NullRange},
      global{static_cast<cl::size_type>(totalsz)},
      local{static_cast<cl::size_type>(localsz)};

  framecl::run_params_t parms{offset, global, local};
  framecl::functor_t<cl::Buffer, cl_int, cl::Buffer, cl::LocalSpaceArg, cl_int>
      histogram(prog, parms, "histogram");

  framecl::buffer_t<cl_uchar> bufs[3] = {
      {ctx, image.data(), totalsz},
      {ctx, image.data() + totalsz, totalsz},
      {ctx, image.data() + 2 * totalsz, totalsz}};
  framecl::buffer_t<cl_int> hist[3] = {
      {ctx, nbins}, {ctx, nbins}, {ctx, nbins}};

  std::vector<framecl::task_t> tasks;
  tasks.reserve(9); // to avoid reallocs and pointer invalidation

  std::vector<std::vector<framecl::task_t *>> dginit;

  for (int i = 0; i < 3; ++i) {
    // write buffer to device
    auto &wtask = tasks.emplace_back(framecl::task::write, bufs[i]);
    // execute
    auto &etask = tasks.emplace_back(framecl::task::process, histogram,
                                     bufs[i].base(), totalsz, hist[i].base(),
                                     cl::Local(nbins * sizeof(int)), nbins);
    // read buffer back
    auto &rtask = tasks.emplace_back(framecl::task::read, hist[i]);
    dginit.emplace_back(std::vector<framecl::task_t *>{&wtask});
    dginit.emplace_back(std::vector<framecl::task_t *>{&etask, &wtask});
    dginit.emplace_back(std::vector<framecl::task_t *>{&rtask, &etask});
  }

  framecl::depgraph_t dg(ctx, dginit.begin(), dginit.end());

  if (opts.verbose()) {
    std::cout << "Dep graph for tasks:" << std::endl;
    dg.dump(std::cout);
    std::cout << std::endl;
  }

  dg.execute();

  auto rhist_max = *std::max_element(hist[0].begin(), hist[0].end());
  auto ghist_max = *std::max_element(hist[1].begin(), hist[1].end());
  auto bhist_max = *std::max_element(hist[2].begin(), hist[2].end());

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
    disp_image(draw_disp_r, hist[0], nbins, rhist_max, red);
    disp_image(draw_disp_g, hist[1], nbins, ghist_max, green);
    disp_image(draw_disp_b, hist[2], nbins, bhist_max, blue);

    // Temporize event loop
    cimg_library::cimg::wait(20);
  }

  return 0;
}

int main(int argc, char **argv) {
  try {
    return histmain(argc, argv);
  } catch (cl::Error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    process_error_t pe(e.err());
    pe(std::cerr);
    std::cerr << std::endl;
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
  return -1;
}