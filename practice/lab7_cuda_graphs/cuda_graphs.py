import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import time
import sys
import os


kernel_stripe_code = """
__global__ void floyd_stripe_k(int *dist, int N, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N && dist[i * N + k] != 2147483647) {
        int dik = dist[i * N + k];
        for (int j = 0; j < N; ++j) {
            int dkj = dist[k * N + j];
            if (dkj != 2147483647) {
                int new_dist = dik + dkj;
                if (new_dist < dist[i * N + j]) {
                    dist[i * N + j] = new_dist;
                }
            }
        }
    }
}
"""

kernel_block_code = """
#define TILE_SIZE 32

__global__ void floyd_block_k(int *dist, int N, int k)
{
    __shared__ int tile_k[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tid = threadIdx.x;
        int j = t * TILE_SIZE + tid;
        
        if (j < N) {
            tile_k[tid] = dist[k * N + j];
        } else {
            tile_k[tid] = 2147483647;
        }
        __syncthreads();
        
        if (i < N && dist[i * N + k] != 2147483647) {
            int dik = dist[i * N + k];
            int start_j = t * TILE_SIZE;
            int end_j = min(start_j + TILE_SIZE, N);
            
            for (int tj = 0; tj < end_j - start_j; ++tj) {
                if (tile_k[tj] != 2147483647) {
                    int new_dist = dik + tile_k[tj];
                    int idx = i * N + start_j + tj;
                    if (new_dist < dist[idx]) {
                        dist[idx] = new_dist;
                    }
                }
            }
        }
    }
}
"""


compile_options = [
    '--allow-unsupported-compiler',
    '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
]

old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

mod_stripe = SourceModule(kernel_stripe_code, options=compile_options)
mod_block = SourceModule(kernel_block_code, options=compile_options)

sys.stderr = old_stderr

kernel_stripe = mod_stripe.get_function("floyd_stripe_k")
kernel_block = mod_block.get_function("floyd_block_k")

INF = 2147483647


def floyd_sequential(dist):
    N = dist.shape[0]
    for k in range(N):
        for i in range(N):
            if dist[i, k] == INF:
                continue
            dik = dist[i, k]
            for j in range(N):
                if dist[k, j] == INF:
                    continue
                new_dist = dik + dist[k, j]
                if new_dist < dist[i, j]:
                    dist[i, j] = new_dist
    return dist

def floyd_stripe_gpu(dist, N):
    dist_gpu = cuda.mem_alloc(dist.nbytes)
    cuda.memcpy_htod(dist_gpu, dist)
    
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    
    for k in range(N):
        kernel_stripe(
            dist_gpu, np.int32(N), np.int32(k),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
    
    cuda.Context.synchronize()
    
    result = np.zeros((N, N), dtype=np.int32)
    cuda.memcpy_dtoh(result, dist_gpu)
    
    return result

def floyd_block_gpu(dist, N, TILE_SIZE=32):
    dist_gpu = cuda.mem_alloc(dist.nbytes)
    cuda.memcpy_htod(dist_gpu, dist)
    
    block_size = TILE_SIZE
    grid_size = (N + block_size - 1) // block_size
    
    for k in range(N):
        kernel_block(
            dist_gpu, np.int32(N), np.int32(k),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
    
    cuda.Context.synchronize()
    
    result = np.zeros((N, N), dtype=np.int32)
    cuda.memcpy_dtoh(result, dist_gpu)
    
    return result

def generate_dense_graph(N):
    dist = np.full((N, N), INF, dtype=np.int32)
    np.fill_diagonal(dist, 0)
    for i in range(N):
        for j in range(N):
            if i != j and np.random.random() < 0.5:
                dist[i, j] = np.random.randint(1, 101)
    return dist

def generate_sparse_graph(N):
    dist = np.full((N, N), INF, dtype=np.int32)
    np.fill_diagonal(dist, 0)
    edges = max(1, N // 10)
    for i in range(N):
        for _ in range(edges):
            j = np.random.randint(0, N)
            if i != j:
                dist[i, j] = np.random.randint(1, 101)
    return dist

def generate_complete_graph(N):
    dist = np.random.randint(1, 101, (N, N), dtype=np.int32)
    np.fill_diagonal(dist, 0)
    return dist

def verify_results(result1, result2):
    return np.array_equal(result1, result2)

def run_benchmark():
    sizes = [100, 200, 400]
    graph_generators = [
        ("Dense (50% edges)", generate_dense_graph),
        ("Sparse (10% edges)", generate_sparse_graph),
        ("Complete (100% edges)", generate_complete_graph)
    ]
    
    print("\n" + "=" * 120)
    print("FLOYD-WARSHALL ALGORITHM: SEQUENTIAL vs PARALLEL (PyCUDA)")
    print("=" * 120)
    
    all_results = []
    
    for graph_name, generator in graph_generators:
        print("\n" + "-" * 120)
        print("GRAPH TYPE: {}".format(graph_name))
        print("-" * 120)
        print("{:<8} {:<20} {:<15} {:<12} {:<15}".format(
            "Size", "Method", "Time (ms)", "Speedup", "Verification"))
        print("-" * 120)
        
        for N in sizes:
            graph = generator(N)
            
            graph_cpu = graph.copy()
            start = time.perf_counter()
            result_cpu = floyd_sequential(graph_cpu)
            cpu_time = (time.perf_counter() - start) * 1000
            
            print("{:<8} {:<20} {:<15.2f} {:<12}".format(
                N, "Sequential (CPU)", cpu_time, "1.00x"))
            
            all_results.append({
                'graph': graph_name,
                'size': N,
                'method': 'Sequential',
                'time': cpu_time,
                'speedup': 1.0
            })
            
            if N <= 400:
                graph_stripe = graph.copy()
                start = time.perf_counter()
                result_stripe = floyd_stripe_gpu(graph_stripe, N)
                stripe_time = (time.perf_counter() - start) * 1000
                stripe_speedup = cpu_time / stripe_time
                verified_stripe = verify_results(result_cpu, result_stripe)
                
                speedup_str = "{:.2f}x".format(stripe_speedup)
                ver_str = "OK" if verified_stripe else "FAIL"
                print("{:<8} {:<20} {:<15.2f} {:<12} {:<15}".format(
                    N, "GPU Stripe", stripe_time, speedup_str, ver_str))
                
                all_results.append({
                    'graph': graph_name,
                    'size': N,
                    'method': 'Stripe',
                    'time': stripe_time,
                    'speedup': stripe_speedup
                })
                
                graph_block = graph.copy()
                start = time.perf_counter()
                result_block = floyd_block_gpu(graph_block, N)
                block_time = (time.perf_counter() - start) * 1000
                block_speedup = cpu_time / block_time
                verified_block = verify_results(result_cpu, result_block)
                
                speedup_str = "{:.2f}x".format(block_speedup)
                ver_str = "OK" if verified_block else "FAIL"
                print("{:<8} {:<20} {:<15.2f} {:<12} {:<15}".format(
                    N, "GPU Block", block_time, speedup_str, ver_str))
                
                all_results.append({
                    'graph': graph_name,
                    'size': N,
                    'method': 'Block',
                    'time': block_time,
                    'speedup': block_speedup
                })
            else:
                print("{:<8} {:<20} {:<15}".format(N, "GPU Stripe", "SKIP (OOM)"))
                print("{:<8} {:<20} {:<15}".format(N, "GPU Block", "SKIP (OOM)"))
        
        print("\nANALYSIS FOR {}:".format(graph_name))
        print("-" * 60)
        for N in sizes:
            graph = generator(N)
            graph_cpu = graph.copy()
            start = time.perf_counter()
            floyd_sequential(graph_cpu)
            cpu_time = (time.perf_counter() - start) * 1000
            
            if N <= 400:
                graph_gpu = graph.copy()
                start = time.perf_counter()
                floyd_block_gpu(graph_gpu, N)
                gpu_time = (time.perf_counter() - start) * 1000
                speedup = cpu_time / gpu_time
                
                msg = "  Size N={}: speedup = {:.2f}x".format(N, speedup)
                if speedup >= 3.5:
                    msg += " (excellent scalability)"
                elif speedup >= 2.5:
                    msg += " (good scalability)"
                else:
                    msg += " (moderate scalability)"
                print(msg)
    
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    header = "{:<8} {:<25} {:<12} {:<18} {:<17} {:<12} {:<12}".format(
        "Size", "Graph Type", "CPU (ms)", "GPU Stripe (ms)", 
        "GPU Block (ms)", "Stripe Spd", "Block Spd")
    print(header)
    print("-" * 120)
    
    for N in sizes:
        for graph_name, _ in graph_generators:
            size_results = [r for r in all_results 
                          if r['size'] == N and r['graph'] == graph_name]
            
            if len(size_results) >= 3:
                cpu = next(r for r in size_results if r['method'] == 'Sequential')
                stripe = next(r for r in size_results if r['method'] == 'Stripe')
                block = next(r for r in size_results if r['method'] == 'Block')
                
                stripe_spd = "{:.2f}x".format(stripe['speedup'])
                block_spd = "{:.2f}x".format(block['speedup'])
                
                print("{:<8} {:<25} {:<12.2f} {:<18.2f} {:<17.2f} {:<12} {:<12}".format(
                    N, graph_name, cpu['time'], stripe['time'], block['time'],
                    stripe_spd, block_spd))
    
    print("=" * 120)


def main():
    print("=" * 120)
    print("FLOYD-WARSHALL ALGORITHM ON GPU")
    print("=" * 120)
    
    device = cuda.Device(0)
    print("\nGPU Information:")
    print("  Device: {}".format(device.name()))
    print("  Compute Capability: {}".format(device.compute_capability()))
    print("  Total Memory: {:.2f} GB".format(device.total_memory() / 1024**3))
    print("  Max Threads per Block: {}".format(device.max_threads_per_block))
    
    run_benchmark()


if __name__ == "__main__":
    main()