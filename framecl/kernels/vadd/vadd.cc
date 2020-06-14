//-----------------------------------------------------------------------------
//
// vadd in framecl
//
//-----------------------------------------------------------------------------
//
// simple example how to use framecl for vadd
//
//-----------------------------------------------------------------------------

#include <numeric>

#include "framecl.hpp"

constexpr int DEFSZ = 100;

int vaddmain(int argc, char **argv) {
  framecl::optparser_t opts;

  // user may add custom options before parse call
  opts.add<int>("size", DEFSZ, "data size");

  // now a ton of useful options registered: device info, platforms, etc
  opts.parse(argc, argv);

  // get option value, use custom checker
  int size = opts.check<int>(
      "size", [](int sz) { return sz > 0; }, "size shall be > 0");

  // hello message
  if (!opts.quiet()) {
    std::cout << "Hello from vadd with size = " << size << std::endl;
    std::cout << "Use --size option to change size" << std::endl;
  }

  // some standard checkers
  opts.require_platform("This program requires platform specification. Use "
                        "--list for available platforms");
  opts.require_program("This program needs external cl program file. It shall "
                       "contain 'vadd' kernel");

  // Context and program
  framecl::context_t ctx(opts);
  framecl::program_t prog(ctx, opts);

  cl::NDRange offset(cl::NullRange), global(DEFSZ), local(cl::NullRange);

  framecl::run_params_t parms{offset, global, local};
  framecl::functor_t<cl::Buffer, cl::Buffer, cl::Buffer> vadd(prog, parms,
                                                              "vector_add");

  // input buffers
  framecl::buffer_t<cl_int> bufA(ctx, size), bufB(ctx, size);
  std::iota(bufA.begin(), bufA.end(), 0);
  std::iota(bufB.rbegin(), bufB.rend(), 0);

  if (opts.verbose()) {
    std::cout << "bufA: ";
    bufA.dump(std::cout);
    std::cout << std::endl;
    std::cout << "bufB: ";
    bufB.dump(std::cout);
    std::cout << std::endl;
  }

  // output buffer
  framecl::buffer_t<cl_int> bufC(ctx, size);

  // tasks
  framecl::task_t writeA(framecl::task::write, bufA);
  framecl::task_t writeB(framecl::task::write, bufB);
  framecl::task_t execF(framecl::task::process, vadd, bufA.base(), bufB.base(),
                        bufC.base());
  framecl::task_t readC(framecl::task::read, bufC);

  // dependency graph to execute as a graph (like DPC++ but in framework, not in
  // language)
  framecl::depgraph_t dg(
      ctx,
      {{&writeA}, {&writeB}, {&execF, &writeA, &writeB}, {&readC, &execF}});

  if (opts.verbose()) {
    std::cout << "Dep graph for tasks:" << std::endl;
    dg.dump(std::cout);
    std::cout << std::endl;
  }

  // do execute
  dg.execute();

  if (opts.verbose()) {
    std::cout << "bufC: ";
    bufC.dump(std::cout);
    std::cout << std::endl;
  }

  if (!opts.quiet())
    std::cout << "Done" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  try {
    return vaddmain(argc, argv);
  } catch (cl::Error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    process_error_t pe(e.err());
    pe(std::cerr);
    std::cerr << std::endl;
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
}
