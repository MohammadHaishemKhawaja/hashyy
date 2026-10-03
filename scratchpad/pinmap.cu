// Can a WRITABLE file mapping be page-locked and DMA'd from directly?
//
// Every L1 miss currently costs: ReadFile (page cache -> pinned staging memcpy) and then a
// DMA that reads those bytes again. That is 3 passes over DRAM per byte delivered. If the
// file's own pages can be registered with CUDA, the memcpy disappears and it is 1 pass.
// cudaHostRegisterReadOnly is unsupported on this WDDM box, but plain writable registration
// reports supported - so map the GGUF read-write (we never write) and try that.
#include <cstdio>
#include <cstdlib>
#include <windows.h>
#include <cuda_runtime.h>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: pinmap <GiB> <path-to-a-large-file>\n");
        return 1;
    }
    const char* path = argv[2];
    size_t want = (argc>1 ? atoll(argv[1]) : 4) * (1ull<<30);

    HANDLE h = CreateFileA(path, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { printf("CreateFile rw failed %lu\n", GetLastError()); return 1; }
    HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!m) { printf("CreateFileMapping rw failed %lu\n", GetLastError()); return 1; }
    void* base = MapViewOfFile(m, FILE_MAP_READ|FILE_MAP_WRITE, 0, 0, want);
    if (!base) { printf("MapViewOfFile failed %lu\n", GetLastError()); return 1; }
    printf("mapped %.1f GiB read-write at %p\n", want/1073741824.0, base);

    // make it resident first; registering a non-resident range is what we want to avoid paying for
    auto t0 = std::chrono::steady_clock::now();
    volatile unsigned char sink = 0;
    for (size_t o = 0; o < want; o += 4096) sink ^= ((unsigned char*)base)[o];
    printf("touched in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    cudaError_t e = cudaHostRegister(base, want, cudaHostRegisterDefault);
    double reg = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    if (e != cudaSuccess) { printf("cudaHostRegister FAILED: %s\n", cudaGetErrorString(e)); return 2; }
    printf("cudaHostRegister OK in %.1f s\n", reg);

    void* dev; cudaMalloc(&dev, 256ull<<20);
    const size_t SZ = 626*1024;
    // DMA straight out of the mapping
    t0 = std::chrono::steady_clock::now(); long long n=0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count() < 2.0) {
        for (int r=0;r<32;r++) {
            size_t off = ((n+r)*7919*SZ) % (want - SZ);
            cudaMemcpyAsync((char*)dev + ((n+r)%256)*SZ, (char*)base+off, SZ, cudaMemcpyHostToDevice, cudaStreamPerThread);
        }
        cudaStreamSynchronize(cudaStreamPerThread);
        n += 32;
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    printf("DMA from the registered mapping: %.2f GB/s\n", n*(double)SZ/dt/1e9);
    cudaHostUnregister(base);
    return 0;
}
