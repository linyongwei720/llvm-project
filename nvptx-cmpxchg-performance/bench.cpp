#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS)
    return;
  const char *message = "unknown";
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "%s: %s\n", operation, message);
  std::exit(1);
}

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  check(cuInit(0), "cuInit");
  CUdevice device;
  check(cuDeviceGet(&device, 0), "device");
  char name[256]{};
  int sm_count = 0;
  check(cuDeviceGetName(name, sizeof(name), device), "device name");
  check(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                            device), "SM count");
  CUcontext context;
  check(cuCtxCreate(&context, nullptr, 0, device), "create context");
  CUmodule module;
  check(cuModuleLoad(&module, argv[1]), "load module");
  CUfunction functions[2];
  check(cuModuleGetFunction(&functions[0], module, "bench_excluded"), "control");
  check(cuModuleGetFunction(&functions[1], module, "bench_guard"), "guard");
  int registers[2], local_bytes[2];
  for (int v = 0; v != 2; ++v) {
    check(cuFuncGetAttribute(&registers[v], CU_FUNC_ATTRIBUTE_NUM_REGS,
                            functions[v]), "registers");
    check(cuFuncGetAttribute(&local_bytes[v], CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            functions[v]), "local size");
  }
  const unsigned max_threads = 128 * 32;
  CUdeviceptr scratch, output;
  check(cuMemAlloc(&scratch, max_threads * sizeof(std::uint32_t)), "scratch");
  check(cuMemAlloc(&output, max_threads * sizeof(std::uint32_t)), "output");
  CUevent begin, end;
  check(cuEventCreate(&begin, 0), "begin event");
  check(cuEventCreate(&end, 0), "end event");
  std::mt19937 rng(240760);
  std::printf("{\"device\":\"%s\",\"SMs\":%d,\"registers_control_guard\":[%d,%d],"
              "\"local_bytes_control_guard\":[%d,%d],\"cases\":[",
              name, sm_count, registers[0], registers[1], local_bytes[0],
              local_bytes[1]);
  bool first_case = true;
  for (unsigned space = 0; space != 2; ++space) {
    for (unsigned mode = 0; mode != 2; ++mode) {
      const unsigned blocks = mode ? 32 : 1;
      const unsigned threads = mode ? 128 : 1;
      unsigned iterations = mode ? 2048 : 32768;
      void *args[] = {&scratch, &output, &iterations, &space};
      auto launch = [&](int variant) {
        check(cuEventRecord(begin, nullptr), "record begin");
        check(cuLaunchKernel(functions[variant], blocks, 1, 1, threads, 1, 1,
                             0, nullptr, args, nullptr), "launch");
        check(cuEventRecord(end, nullptr), "record end");
        check(cuEventSynchronize(end), "synchronize event");
        float ms = 0;
        check(cuEventElapsedTime(&ms, begin, end), "elapsed time");
        std::vector<std::uint32_t> actual(blocks * threads);
        check(cuMemcpyDtoH(actual.data(), output, actual.size() * sizeof(actual[0])),
              "copy output");
        if (!std::all_of(actual.begin(), actual.end(),
                         [&](std::uint32_t value) { return value == iterations; })) {
          std::fprintf(stderr, "Incorrect benchmark result\n");
          std::exit(2);
        }
        return double(ms);
      };
      for (unsigned warmup = 0; warmup != 6; ++warmup) {
        launch(0);
        launch(1);
      }
      std::vector<double> controls, guards, ratios;
      struct Sample { double control, guard; int first; };
      std::vector<Sample> samples;
      for (unsigned repeat = 0; repeat != 21; ++repeat) {
        const int first = rng() & 1;
        double elapsed[2];
        elapsed[first] = launch(first);
        elapsed[1 - first] = launch(1 - first);
        controls.push_back(elapsed[0]);
        guards.push_back(elapsed[1]);
        ratios.push_back(elapsed[1] / elapsed[0]);
        samples.push_back({elapsed[0], elapsed[1], first});
      }
      std::sort(ratios.begin(), ratios.end());
      if (!first_case)
        std::printf(",");
      first_case = false;
      std::printf("{\"space\":\"%s\",\"mode\":\"%s\",\"blocks\":%u,"
                  "\"threads_per_block\":%u,\"iterations\":%u,"
                  "\"control_median_ms\":%.9g,\"guard_median_ms\":%.9g,"
                  "\"paired_median_change_pct\":%.6f,"
                  "\"paired_p10_change_pct\":%.6f,\"paired_p90_change_pct\":%.6f,"
                  "\"samples\":[",
                  space ? "shared" : "global", mode ? "parallel" : "serial",
                  blocks, threads, iterations, median(controls), median(guards),
                  (median(ratios) - 1) * 100, (ratios[2] - 1) * 100,
                  (ratios[18] - 1) * 100);
      for (unsigned i = 0; i != samples.size(); ++i) {
        if (i)
          std::printf(",");
        std::printf("{\"control_ms\":%.9g,\"guard_ms\":%.9g,\"first\":%d}",
                    samples[i].control, samples[i].guard, samples[i].first);
      }
      std::printf("]}");
      std::fflush(stdout);
    }
  }
  std::printf("]}\n");
  check(cuEventDestroy(begin), "destroy begin");
  check(cuEventDestroy(end), "destroy end");
  check(cuMemFree(scratch), "free scratch");
  check(cuMemFree(output), "free output");
  check(cuModuleUnload(module), "unload");
  check(cuCtxDestroy(context), "destroy context");
}
