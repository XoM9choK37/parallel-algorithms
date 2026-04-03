#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace chrono;

const int MAX_THREADS = 4;
const int SEQUENTIAL_THRESHOLD = 1000;

void bubble_sort_seq(vector<int>& arr) {
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n-1; ++i) {
        swapped = false;
        for (int j = 0; j < n-i-1; ++j) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

void bubble_sort_par(vector<int>& arr, int num_threads) {
    omp_set_num_threads(num_threads);
    int n = arr.size();
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        #pragma omp parallel for
        for (int i = 0; i < n-1; i += 2) {
            if (arr[i] > arr[i+1]) {
                swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }
        #pragma omp parallel for
        for (int i = 1; i < n-1; i += 2) {
            if (arr[i] > arr[i+1]) {
                swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }
    }
}

void shell_sort_seq(vector<int>& arr) {
    int n = arr.size();
    for (int gap = n/2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j = i;
            while (j >= gap && arr[j - gap] > temp) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

void shell_sort_par(vector<int>& arr, int num_threads) {
    omp_set_num_threads(num_threads);
    int n = arr.size();
    for (int gap = n/2; gap > 0; gap /= 2) {
        #pragma omp parallel for
        for (int k = 0; k < gap; ++k) {
            for (int i = k + gap; i < n; i += gap) {
                int temp = arr[i];
                int j = i;
                while (j >= gap && arr[j - gap] > temp) {
                    arr[j] = arr[j - gap];
                    j -= gap;
                }
                arr[j] = temp;
            }
        }
    }
}

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i+1], arr[high]);
    return i+1;
}

void quick_sort_seq(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort_seq(arr, low, pi-1);
        quick_sort_seq(arr, pi+1, high);
    }
}

void quick_sort_par(vector<int>& arr, int low, int high, int threshold) {
    if (low < high) {
        if (high - low < threshold) {
            quick_sort_seq(arr, low, high);
            return;
        }
        int pi = partition(arr, low, high);
        #pragma omp task
        quick_sort_par(arr, low, pi-1, threshold);
        #pragma omp task
        quick_sort_par(arr, pi+1, high, threshold);
        #pragma omp taskwait
    }
}

void quick_sort_par_wrapper(vector<int>& arr, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp single
        quick_sort_par(arr, 0, arr.size()-1, SEQUENTIAL_THRESHOLD);
    }
}

double measure_time_seq(void (*sort_func)(vector<int>&), vector<int> arr) {
    auto start = high_resolution_clock::now();
    sort_func(arr);
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

double measure_time_par(void (*sort_func)(vector<int>&, int), vector<int> arr, int num_threads) {
    auto start = high_resolution_clock::now();
    sort_func(arr, num_threads);
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

double measure_quick_seq(vector<int> arr) {
    auto start = high_resolution_clock::now();
    quick_sort_seq(arr, 0, arr.size()-1);
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

double measure_quick_par(vector<int> arr, int num_threads) {
    auto start = high_resolution_clock::now();
    quick_sort_par_wrapper(arr, num_threads);
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

void run_benchmark() {
    vector<int> sizes = {10'000, 50'000, 100'000, 500'000};
    vector<int> threads = {2, 3, 4};

    cout << "\n==================== BUBBLE SORT ====================\n";
    cout << setw(10) << "Size" << setw(15) << "Threads" << setw(20) << "Seq (ms)" << setw(20) << "Par (ms)" << setw(15) << "Speedup" << endl;
    for (int n : sizes) {
        vector<int> base(n);
        for (int i = 0; i < n; ++i) base[i] = rand() % 100000;
        double time_seq = measure_time_seq(bubble_sort_seq, base);
        for (int t : threads) {
            double time_par = measure_time_par(bubble_sort_par, base, t);
            double speedup = time_seq / time_par;
            cout << setw(10) << n << setw(15) << t << setw(20) << time_seq << setw(20) << time_par << setw(15) << fixed << setprecision(2) << speedup << endl;
        }
        cout << "---------------------------------------------------------------\n";
    }

    cout << "\n==================== SHELL SORT ====================\n";
    cout << setw(10) << "Size" << setw(15) << "Threads" << setw(20) << "Seq (ms)" << setw(20) << "Par (ms)" << setw(15) << "Speedup" << endl;
    for (int n : sizes) {
        vector<int> base(n);
        for (int i = 0; i < n; ++i) base[i] = rand() % 100000;
        double time_seq = measure_time_seq(shell_sort_seq, base);
        for (int t : threads) {
            double time_par = measure_time_par(shell_sort_par, base, t);
            double speedup = time_seq / time_par;
            cout << setw(10) << n << setw(15) << t << setw(20) << time_seq << setw(20) << time_par << setw(15) << fixed << setprecision(2) << speedup << endl;
        }
        cout << "---------------------------------------------------------------\n";
    }

    cout << "\n==================== QUICK SORT ====================\n";
    cout << setw(10) << "Size" << setw(15) << "Threads" << setw(20) << "Seq (ms)" << setw(20) << "Par (ms)" << setw(15) << "Speedup" << endl;
    for (int n : sizes) {
        vector<int> base(n);
        for (int i = 0; i < n; ++i) base[i] = rand() % 100000;
        double time_seq = measure_quick_seq(base);
        for (int t : threads) {
            double time_par = measure_quick_par(base, t);
            double speedup = time_seq / time_par;
            cout << setw(10) << n << setw(15) << t << setw(20) << time_seq << setw(20) << time_par << setw(15) << fixed << setprecision(2) << speedup << endl;
        }
        cout << "---------------------------------------------------------------\n";
    }
}

int main() {
    srand(time(nullptr));
    run_benchmark();
    return 0;
}