#include <iostream>
#include <string>

#include "framecl.hpp"

// clang-format off
#define cimg_use_jpeg
#include "CImg.h"
// clang-format on

constexpr int DEFSIZE = 1024;
constexpr int DEFBINS = 256;

int histmain(int argc, char **argv) {
  framecl::optparser_t opts;

  opts.add<int>("size", DEFSIZE, "size of source array for histogram");
  opts.add<int>("nbins", DEFBINS, "number of bins in histogram");

  opts.parse(argc, argv);

  opts.require_platform("This program requires platform specification. Use "
                        "--list for available platforms");
  opts.require_program("This program needs external cl program file. It shall "
                       "contain 'histogram' kernel");

  int nbins = opts.check<int>(
      "nbins", [](int nbins) { return nbins > 0; },
      "number of bins shall be > 0");
  int totalsz = opts.check<int>(
      "size", [](int totalsz) { return totalsz > 0; },
      "size fo array shall be > 0");

  if (!opts.quiet()) {
    std::cout << "Hello from hist with nbins = " << nbins << std::endl;
    std::cout << "Use --size option to change source array size" << std::endl;
    std::cout << "Use --nbins option to change number of bins" << std::endl;
  }

  // here most interesting customization point goes
  cl::size_type globalsz = totalsz;
  cl::size_type localsz = nbins;

  framecl::context_t ctx(opts);
  framecl::program_t prog(ctx, opts);

  cl::NDRange offset{cl::NullRange}, global{globalsz}, local{localsz};

  framecl::run_params_t parms{offset, global, local};
  framecl::functor_t<cl::Buffer, cl_int, cl::Buffer, cl::LocalSpaceArg, cl_int>
      histogram(prog, parms, "histogram");

  framecl::buffer_t<cl_uchar> buf{ctx, totalsz};
  framecl::buffer_t<cl_int> hist{ctx, nbins};
  std::vector<int> ref(nbins);

  for (int i = 0; i < totalsz; ++i)
    buf[i] = i % 256;

  if (opts.check())
    for (int i = 0; i < totalsz; ++i)
      ref[buf[i]] += 1;

  framecl::task_t writeBuf(framecl::task::write, buf);
  framecl::task_t execF(framecl::task::process, histogram, buf.base(), totalsz,
                        hist.base(), cl::Local(nbins * sizeof(int)), nbins);
  framecl::task_t readHist(framecl::task::read, hist);

  // clang-format off
  framecl::depgraph_t dg(ctx, {
    {&writeBuf},
    {&execF, &writeBuf},
    {&readHist, &execF}
  });
  // clang-format on

  if (opts.verbose()) {
    std::cout << "Dep graph for tasks:" << std::endl;
    dg.dump(std::cout);
    std::cout << std::endl;
  }

  dg.execute();

  if (opts.check()) {
    std::cout << "Cross-check with non-ocl results... ";
    for (decltype(nbins) i = 0; i < nbins; ++i)
      if (hist[i] != ref[i]) {
        std::cout << "failed at i = " << i << ", hist[i] = " << hist[i]
                  << ", ref[i] = " << ref[i] << std::endl;
        throw std::logic_error("Check failed");
      }
    std::cout << "ok" << std::endl;
  }

  if (!opts.quiet())
    std::cout << "Done" << std::endl;

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