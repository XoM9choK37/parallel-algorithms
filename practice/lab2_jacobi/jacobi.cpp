#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace chrono;

const double EPSILON = 1e-6;
const int MAX_ITER = 1000;

void initialize_system(vector<vector<double>>& A, vector<double>& b, vector<double>& x, int n) {
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            A[i][j] = (i == j) ? n + 1 : (rand() % 10) / 10.0;
            sum += abs(A[i][j]);
        }
        b[i] = sum * (rand() % 5 + 1);
        x[i] = 0.0;
    }
}

double sequential_jacobi(const vector<vector<double>>& A, const vector<double>& b, 
                         vector<double>& x, int n) {
    vector<double> x_new(n, 0.0);
    int iter = 0;
    double error = 0.0;
    
    auto start = high_resolution_clock::now();
    
    do {
        error = 0.0;
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
            error += pow(x_new[i] - x[i], 2);
        }
        error = sqrt(error);
        x.swap(x_new);
        iter++;
    } while (error > EPSILON && iter < MAX_ITER);
    
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

double parallel_jacobi(const vector<vector<double>>& A, const vector<double>& b, 
                       vector<double>& x, int n, int num_threads) {
    vector<double> x_new(n, 0.0);
    int iter = 0;
    double error = 0.0;
    
    omp_set_num_threads(num_threads);
    
    auto start = high_resolution_clock::now();
    
    do {
        error = 0.0;
        
        #pragma omp parallel for reduction(+:error)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
            error += pow(x_new[i] - x[i], 2);
        }
        
        error = sqrt(error);
        x.swap(x_new);
        iter++;
    } while (error > EPSILON && iter < MAX_ITER);
    
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

void run_benchmark(int n, int num_threads) {
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);
    vector<double> x_seq(n);
    vector<double> x_par(n);
    
    initialize_system(A, b, x_seq, n);
    x_par = x_seq;
    
    double time_seq = sequential_jacobi(A, b, x_seq, n);
    double time_par = parallel_jacobi(A, b, x_par, n, num_threads);
    
    cout << setw(10) << n << setw(15) << num_threads 
         << setw(25) << fixed << setprecision(2) << time_seq << " ms"
         << setw(25) << time_par << " ms" << endl;
}

int main() {
    vector<int> sizes = {256, 512, 1024, 2048, 4096};
    vector<int> threads = {1, 2, 3, 4};
    
    cout << "==========================================================================================" << endl;
    cout << setw(10) << "Size" << setw(15) << "Threads" << setw(25) << "Sequential (ms)"
         << setw(25) << "Parallel (ms)" << endl;
    cout << "==========================================================================================" << endl;
    
    for (int n : sizes) {
        for (int t : threads) {
            run_benchmark(n, t);
        }
        cout << "------------------------------------------------------------------------------------------" << endl;
    }
    
    return 0;
}
