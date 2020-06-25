//-----------------------------------------------------------------------------
//
// reduce in framecl
//
//-----------------------------------------------------------------------------
//
// simple example how to use framecl for reduce
//
//-----------------------------------------------------------------------------

#include "framecl.hpp"

constexpr int DEFSZ = 256 * 16;

int reduce_main(int argc, char **argv) {
  framecl::optparser_t opts;

  opts.add<int>("size", DEFSZ, "data size");

  opts.parse(argc, argv);

  int size = opts.check<int>(
      "size", [](int sz) { return sz > 0; }, "size shall be > 0");

  opts.require_platform("This program requires platform specification. Use "
                        "--list for available platforms");
  opts.require_program("This program needs external cl program file. It shall "
                       "contain 'reduce' kernel");

  framecl::context_t ctx(opts);
  int wgsz = ctx.max_workgroup_size();
  int outsz = size / wgsz;

  if (!opts.quiet()) {
    std::cout << "Hello from reduce with size = " << size << std::endl;
    std::cout << "Workgroup size selected = " << wgsz << std::endl;
  }

  framecl::program_t prog(ctx, opts);

  cl::size_type globalsz = size;
  cl::size_type localsz = wgsz;

  cl::NDRange offset{cl::NullRange}, global{globalsz}, local{localsz};
  framecl::run_params_t parms{offset, global, local};
  framecl::functor_t<cl::Buffer, cl::Buffer, cl::LocalSpaceArg> reduce(
      prog, parms, "reduce");

  framecl::buffer_t<cl_int> bufIn{ctx, size}, bufOut{ctx, outsz};
  rand_init(bufIn, size, 0, 10);

  framecl::task_t writeIn(framecl::task::write, bufIn);
  framecl::task_t execF(framecl::task::process, reduce, bufIn.base(),
                        bufOut.base(), cl::Local(localsz * sizeof(int)));
  framecl::task_t readOut(framecl::task::read, bufOut);

  // clang-format off
  framecl::depgraph_t dg(ctx, {
    {&writeIn},
    {&execF, &writeIn},
    {&readOut, &execF}
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
    int refsum = 0, checksum = 0;
    for (auto a : bufIn)
      refsum += a;
    for (auto a : bufOut)
      checksum += a;

    if (refsum != checksum) {
      std::cout << "failed with ocl result = " << checksum
                << ", reference = " << refsum << std::endl;
      throw std::logic_error("Check failed");
    }
    std::cout << "ok" << std::endl;
  }
  return 0;
}

int main(int argc, char **argv) {
  try {
    return reduce_main(argc, argv);
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
