//
//  CUDAArchitectures.hpp
//  Cyanescent Forge — Linux CUDA worker
//
//  Whether a build's code list (FORGE_CUDA_ARCHS, e.g. "sm_86 sm_89
//  compute_89") can run on a device of a given compute capability: SASS runs
//  on the same major version with an equal or higher minor version; PTX is
//  JIT-compiled by the driver for any device of that version or newer.
//  Plain C++ so the host self-tests cover it.
//
#ifndef FORGE_CUDA_ARCHITECTURES_HPP
#define FORGE_CUDA_ARCHITECTURES_HPP

#include <cstdlib>
#include <sstream>
#include <string>

namespace forge {

inline bool cudaArchitecturesCover(const std::string &architectures, int major, int minor)
{
    std::istringstream archs(architectures);
    std::string arch;
    const int device = major * 10 + minor;
    while (archs >> arch) {
        const bool sass = arch.rfind("sm_", 0) == 0;
        const bool ptx = arch.rfind("compute_", 0) == 0;
        if (!sass && !ptx) continue;
        const int version = std::atoi(arch.c_str() + (sass ? 3 : 8));
        if (version <= 0) continue;
        if (sass && version / 10 == major && version % 10 <= minor) return true;
        if (ptx && version <= device) return true;
    }
    return false;
}

} // namespace forge

#endif // FORGE_CUDA_ARCHITECTURES_HPP
