//-----------------------------------------------------------------------------
//
// vadd in framecl
//
//-----------------------------------------------------------------------------
//
// have two options:
//   --size for array size
//   --globalsize for size of global wg
// enqueues a number of parallel kernels, collects execution time
//
//-----------------------------------------------------------------------------

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>

#include "framecl.hpp"

constexpr int DEFSLC = 1;
constexpr int DEFSZ = 100000;
constexpr int DEFGSZ = 10000;

int vaddmain(int argc, char **argv) {
  framecl::optparser_t opts;

  opts.add<int>("size", DEFSZ, "data size");
  opts.add<int>("globalsize", DEFGSZ, "global size");

  opts.parse(argc, argv);

  int size = opts.check<int>(
      "size", [](int sz) { return sz > 0; }, "size shall be > 0");
  int globalsize = opts.check<int>(
      "globalsize", [](int gsz) { return gsz > 0; },
      "global size shall be > 0");
  int nslices = size / globalsize;

  if (!opts.quiet()) {
    std::cout << "Hello from vadd with: " << std::endl
              << "* size = " << size << std::endl
              << "* nslices = " << nslices << std::endl
              << "* globalsize = " << globalsize << std::endl
              << std::endl;
  }

  opts.require_platform("This program requires platform specification. Use "
                        "--list for available platforms");
  opts.require_program("This program needs external cl program file. It shall "
                       "contain 'vector_add' kernel");

  framecl::context_t ctx(opts);
  framecl::program_t prog(ctx, opts);

  std::vector<cl_int> VA(size), VB(size), VC(size);
  std::iota(VA.begin(), VA.end(), 0);
  std::iota(VB.rbegin(), VB.rend(), 0);

  std::vector<framecl::task_t> tasks;
  tasks.reserve(nslices * 4); // to avoid reallocs and pointer invalidation
                              // 4 is for 2 writes, 1 exec, 1 read

  std::vector<framecl::buffer_t<cl_int>> bufs;
  bufs.reserve(nslices * 3);

  std::vector<std::vector<framecl::task_t *>> dginit;

  cl::size_type gsz = globalsize;
  cl::NDRange offset{cl::NullRange}, global{gsz}, local{cl::NullRange};

  framecl::run_params_t parms{offset, global, local};
  framecl::functor_t<cl::Buffer, cl::Buffer, cl::Buffer> vadd(prog, parms,
                                                              "vector_add");
  std::unordered_map<framecl::task_t *, std::string> tnames;

  int last_size = globalsize;

  if ((size % nslices) != 0) {
    last_size = size % nslices;
    nslices += 1;
  }

  int *vaptr = VA.data();
  int *vbptr = VB.data();
  int *vcptr = VC.data();

  for (int slice = 0; slice < nslices; ++slice) {
    int internal_size = globalsize;

    // possible "tail" situation
    if (slice == nslices - 1)
      internal_size = last_size;

    auto &bufA = bufs.emplace_back(ctx, vaptr, internal_size);
    auto &bufB = bufs.emplace_back(ctx, vbptr, internal_size);
    auto &bufC = bufs.emplace_back(ctx, vcptr, internal_size);

    vaptr += internal_size;
    vbptr += internal_size;
    vcptr += internal_size;

    // TODO: I can see here problem: buffers live too long
    //       Really here shall go enqueue map/unmap commands to explicitly
    //       unmap?
    auto &wtaskA = tasks.emplace_back(framecl::task::write, bufA);
    auto &wtaskB = tasks.emplace_back(framecl::task::write, bufB);
    auto &etask = tasks.emplace_back(framecl::task::process, vadd, bufA.base(),
                                     bufB.base(), bufC.base());
    auto &rtaskC = tasks.emplace_back(framecl::task::read, bufC);

    tnames[&wtaskA] = "Buf write A";
    tnames[&wtaskB] = "Buf write B";
    tnames[&etask] = "Execute";
    tnames[&rtaskC] = "Buf read C";

    // TODO: problem here as well -- wtaskA shall depend on unmap
    dginit.emplace_back(std::vector<framecl::task_t *>{&wtaskA});
    dginit.emplace_back(std::vector<framecl::task_t *>{&wtaskB});
    dginit.emplace_back(
        std::vector<framecl::task_t *>{&etask, &wtaskA, &wtaskB});
    dginit.emplace_back(std::vector<framecl::task_t *>{&rtaskC, &etask});
  }

  framecl::depgraph_t dg(ctx, dginit.begin(), dginit.end());

  if (opts.verbose()) {
    std::cout << "Dep graph for tasks:" << std::endl;
    dg.dump(std::cout);
    std::cout << std::endl;
  }

  // do execute
  dg.execute();

  if (!opts.quiet()) {
    std::cout << "Total elapsed time: " << dg.elapsed() << std::endl;
    for (auto &tsk : tasks) {
      std::cout << tnames[&tsk] << " elapsed time: " << dg.task_elapsed(&tsk)
                << std::endl;
    }
  }

  if (opts.check()) {
    std::cout << "Cross-check with non-ocl results... ";
    for (decltype(size) i = 0; i < size; ++i)
      if (VA[i] + VB[i] != VC[i]) {
        std::cout << "failed at i = " << i << ", bufA[i] = " << VA[i]
                  << ", bufB[i] = " << VB[i] << ", bufC[i] = " << VC[i]
                  << std::endl;
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
    return vaddmain(argc, argv);
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
