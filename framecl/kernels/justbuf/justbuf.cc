//-----------------------------------------------------------------------------
//
// justbuf in framecl
//
//-----------------------------------------------------------------------------
//
// simple example of memory writing and reading
//
//-----------------------------------------------------------------------------

#include <numeric>

#include "framecl.hpp"

constexpr int DEFSZ = 100;

int justbufmain(int argc, char **argv) {
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
    std::cout << "Hello from justbuf with size = " << size << std::endl;
    std::cout << "Use --size option to change size" << std::endl;
  }

  // some standard checkers
  opts.require_platform("This program requires platform specification. Use "
                        "--list for available platforms");

  // Context and program
  framecl::context_t ctx(opts);

  // input-output buffer
  framecl::buffer_t<cl_int> bufA(ctx, size);
  std::iota(bufA.begin(), bufA.end(), 0);

  if (opts.verbose()) {
    std::cout << "bufA: ";
    bufA.dump(std::cout);
    std::cout << std::endl;
  }

  // reference
  std::vector ref(bufA.begin(), bufA.end());

  // tasks
  framecl::task_t writeA(framecl::task::write, bufA);
  framecl::task_t readA(framecl::task::read, bufA);

  // dependency graph to execute as a graph (like DPC++ but in framework, not in
  // language)
  framecl::depgraph_t dg(ctx, {{&writeA}, {&readA, &writeA}});

  if (opts.verbose()) {
    std::cout << "Dep graph for tasks:" << std::endl;
    dg.dump(std::cout);
    std::cout << std::endl;
  }

  // do execute
  dg.execute();

  if (opts.verbose()) {
    std::cout << "bufA: ";
    bufA.dump(std::cout);
    std::cout << std::endl;
  }

  if (!opts.quiet())
    std::cout << "Done" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  try {
    return justbufmain(argc, argv);
  } catch (cl::Error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    process_error_t pe(e.err());
    pe(std::cerr);
    std::cerr << std::endl;
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
}
