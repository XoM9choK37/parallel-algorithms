#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace chrono;

void initialize_matrices(vector<vector<double>>& A, vector<vector<double>>& B, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }
}

void sequential_multiply(const vector<vector<double>>& A, 
                          const vector<vector<double>>& B, 
                          vector<vector<double>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallel_strip_multiply(const vector<vector<double>>& A, 
                              const vector<vector<double>>& B, 
                              vector<vector<double>>& C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallel_block_multiply(const vector<vector<double>>& A, 
                              const vector<vector<double>>& B, 
                              vector<vector<double>>& C, int n, int block_size = 64) {
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                for (int i = ii; i < min(ii + block_size, n); ++i) {
                    for (int j = jj; j < min(jj + block_size, n); ++j) {
                        double sum = 0.0;
                        for (int k = kk; k < min(kk + block_size, n); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        #pragma omp atomic
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

void run_benchmark(int n, int num_threads) {
    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C_seq(n, vector<double>(n, 0.0));
    vector<vector<double>> C_strip(n, vector<double>(n, 0.0));
    vector<vector<double>> C_block(n, vector<double>(n, 0.0));

    initialize_matrices(A, B, n);
    omp_set_num_threads(num_threads);

    if (num_threads == 1) {
        auto start = high_resolution_clock::now();
        sequential_multiply(A, B, C_seq, n);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        cout << setw(10) << n << setw(15) << num_threads << setw(25) << duration << " ms";
    }
    
    fill(C_strip.begin(), C_strip.end(), vector<double>(n, 0.0));
    auto start_strip = high_resolution_clock::now();
    parallel_strip_multiply(A, B, C_strip, n);
    auto end_strip = high_resolution_clock::now();
    auto duration_strip = duration_cast<milliseconds>(end_strip - start_strip).count();

    fill(C_block.begin(), C_block.end(), vector<double>(n, 0.0));
    auto start_block = high_resolution_clock::now();
    parallel_block_multiply(A, B, C_block, n);
    auto end_block = high_resolution_clock::now();
    auto duration_block = duration_cast<milliseconds>(end_block - start_block).count();

    if (num_threads == 1) {
        cout << setw(25) << duration_strip << " ms" << setw(25) << duration_block << " ms" << endl;
    } else {
        cout << setw(10) << n << setw(15) << num_threads 
             << setw(25) << "-" << setw(25) << duration_strip << " ms" 
             << setw(25) << duration_block << " ms" << endl;
    }
}

int main() {
    vector<int> sizes = {256, 512, 1024};
    vector<int> threads = {1, 2, 3, 4};

    cout << "=======================================================================================================" << endl;
    cout << setw(10) << "Size" << setw(15) << "Threads" << setw(25) << "Sequential (ms)"
         << setw(25) << "Strip (ms)" << setw(25) << "Block (ms)" << endl;
    cout << "=======================================================================================================" << endl;

    for (int n : sizes) {
        for (int t : threads) {
            run_benchmark(n, t);
        }
        cout << "-------------------------------------------------------------------------------------------------------" << endl;
    }
    
    return 0;
}
