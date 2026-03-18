#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace chrono;

double f(double x) {
    return sin(x) * cos(x) + x * x;
}

double sequential_simpson(double a, double b, int n) {
    if (n % 2 != 0) n++;
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * f(x);
        } else {
            sum += 4 * f(x);
        }
    }
    
    return (h / 3) * sum;
}

double parallel_simpson(double a, double b, int n, int num_threads) {
    if (n % 2 != 0) n++;
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        double local_sum = 0.0;
        
        #pragma omp for
        for (int i = 1; i < n; i++) {
            double x = a + i * h;
            if (i % 2 == 0) {
                local_sum += 2 * f(x);
            } else {
                local_sum += 4 * f(x);
            }
        }
        
        #pragma omp atomic
        sum += local_sum;
    }
    
    return (h / 3) * sum;
}

double measure_time(double a, double b, int n, bool parallel, int threads = 1) {
    if (parallel) {
        parallel_simpson(a, b, n, threads);
    } else {
        sequential_simpson(a, b, n);
    }
    
    auto start = high_resolution_clock::now();
    
    double result;
    int iterations = 5;
    
    for (int it = 0; it < iterations; it++) {
        if (parallel) {
            result = parallel_simpson(a, b, n, threads);
        } else {
            result = sequential_simpson(a, b, n);
        }
        if (result < 0) {
            cout << result;
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count() / iterations;
    
    return duration / 1000.0;
}

int main() {
    double a = 0.0;
    double b = M_PI;
    vector<int> n_values = {1000, 10000, 100000, 1000000, 10000000};
    vector<int> thread_counts = {2, 3, 4};
    
    cout << "==================================================================================================" << endl;
    cout << setw(12) << "N (points)" 
         << setw(15) << "Threads"
         << setw(20) << "Sequential (ms)"
         << setw(20) << "Parallel (ms)"
         << setw(15) << "Speedup" << endl;
    cout << "==================================================================================================" << endl;
    
    for (int n : n_values) {
        double time_seq = measure_time(a, b, n, false);
        double seq_result = sequential_simpson(a, b, n);
        
        for (int t : thread_counts) {
            double time_par = measure_time(a, b, n, true, t);
            double par_result = parallel_simpson(a, b, n, t);
            double speedup = time_seq / time_par;
            
            cout << setw(12) << n 
                 << setw(15) << t
                 << setw(20) << fixed << setprecision(2) << time_seq
                 << setw(20) << time_par
                 << setw(15) << setprecision(2) << speedup << endl;
        }
        cout << "--------------------------------------------------------------------------------------------------" << endl;
    }
    
    cout << "\nТочность вычислений:" << endl;
    cout << "Точное значение: " << fixed << setprecision(8) 
         << (M_PI*M_PI*M_PI/3) << endl;
    cout << "Последовательно: " << sequential_simpson(a, b, 10000000) << endl;
    cout << "Параллельно (4 ядра): " << parallel_simpson(a, b, 10000000, 4) << endl;
    
    return 0;
}
