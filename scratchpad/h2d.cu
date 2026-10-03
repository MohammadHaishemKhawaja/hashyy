// How much H2D bandwidth can N worker threads actually get with cudaStreamPerThread
// and ~626 KiB pinned transfers - the exact shape of an expert-slab upload.
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

int main(int argc, char** argv) {
    size_t sz = argc>1 ? atoll(argv[1]) : 626*1024;
    double secs = argc>2 ? atof(argv[2]) : 2.0;
    void* dev; CK(cudaMalloc(&dev, 512ull<<20));
    printf("transfer %zu KiB, %.1fs per point\n", sz/1024, secs);
    for (int nthr : {1,2,4,6,10,16}) {
        std::vector<void*> hp(nthr);
        for (int i=0;i<nthr;i++) CK(cudaHostAlloc(&hp[i], sz, cudaHostAllocDefault));
        std::atomic<long long> bytes{0};
        auto t0=std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (int i=0;i<nthr;i++) ts.emplace_back([&,i]{
            long long n=0;
            while (std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count() < secs) {
                for (int r=0;r<16;r++) {
                    cudaMemcpyAsync((char*)dev + ((n+r)%256)*sz, hp[i], sz, cudaMemcpyHostToDevice, cudaStreamPerThread);
                    cudaStreamSynchronize(cudaStreamPerThread);
                }
                n+=16;
            }
            bytes += n*(long long)sz;
        });
        for (auto&t:ts) t.join();
        double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        printf("  sync-each  threads %2d : %6.2f GB/s\n", nthr, bytes/dt/1e9);
        // same but one sync per 16 copies
        bytes=0; t0=std::chrono::steady_clock::now(); ts.clear();
        for (int i=0;i<nthr;i++) ts.emplace_back([&,i]{
            long long n=0;
            while (std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count() < secs) {
                for (int r=0;r<16;r++)
                    cudaMemcpyAsync((char*)dev + ((n+r)%256)*sz, hp[i], sz, cudaMemcpyHostToDevice, cudaStreamPerThread);
                cudaStreamSynchronize(cudaStreamPerThread);
                n+=16;
            }
            bytes += n*(long long)sz;
        });
        for (auto&t:ts) t.join();
        dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        printf("  sync-batch threads %2d : %6.2f GB/s\n", nthr, bytes/dt/1e9);
        for (int i=0;i<nthr;i++) cudaFreeHost(hp[i]);
    }
    return 0;
}
