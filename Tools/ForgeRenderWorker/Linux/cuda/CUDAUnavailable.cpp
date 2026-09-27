//
//  CUDAUnavailable.cpp
//  Cyanescent Forge — Linux CUDA worker
//
//  Linked instead of CUDABackend.cu when the worker is built without nvcc
//  (FORGE_BUILD_WITHOUT_CUDA=1: CI and development machines without the CUDA
//  toolkit). The worker then reports that CUDA is not compiled in and can
//  only run the CPU emulation backend; it never pretends to have a GPU.
//
#include "../src/GPUBackend.hpp"

namespace forge {

bool cudaCompiledIn() { return false; }

std::string cudaVersionString(int version)
{
    if (version <= 0) return "none";
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

CUDAEnvironment cudaEnvironment()
{
    CUDAEnvironment env;
    env.error = "This ForgeRenderWorker was built without CUDA (FORGE_BUILD_WITHOUT_CUDA=1 or no nvcc). "
                "Rebuild with the CUDA toolkit: Tools/ForgeRenderWorker/build-linux-cuda.sh.";
    return env;
}

void cudaQueryFreeMemory(std::vector<CUDADeviceInfo> &) {}

std::unique_ptr<RenderBackend> makeCUDABackend(const BackendOptions &)
{
    throw BackendError(cudaEnvironment().error);
}

} // namespace forge
