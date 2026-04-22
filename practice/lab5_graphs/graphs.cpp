#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include <climits>
#include <functional>

using namespace std;
using namespace chrono;

const int INF = INT_MAX / 2;
const int BLOCK_SIZE = 64;

void floyd_warshall_seq(vector<vector<int>>& dist) {
    int n = dist.size();
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            if (dist[i][k] == INF) continue;
            for (int j = 0; j < n; ++j) {
                if (dist[k][j] == INF) continue;
                int new_dist = dist[i][k] + dist[k][j];
                if (new_dist < dist[i][j]) {
                    dist[i][j] = new_dist;
                }
            }
        }
    }
}

void floyd_warshall_par_blocked(vector<vector<int>>& dist, int num_threads) {
    omp_set_num_threads(num_threads);
    int n = dist.size();
    int block_size = BLOCK_SIZE;
    
    for (int k_block = 0; k_block < n; k_block += block_size) {
        int k_end = min(k_block + block_size, n);
        
        for (int k = k_block; k < k_end; ++k) {
            #pragma omp parallel for schedule(static)
            for (int i = k_block; i < k_end; ++i) {
                if (dist[i][k] == INF) continue;
                for (int j = k_block; j < k_end; ++j) {
                    if (dist[k][j] == INF) continue;
                    int new_dist = dist[i][k] + dist[k][j];
                    if (new_dist < dist[i][j]) {
                        dist[i][j] = new_dist;
                    }
                }
            }
        }
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i_block = 0; i_block < n; i_block += block_size) {
                    if (i_block == k_block) continue;
                    int i_end = min(i_block + block_size, n);
                    
                    #pragma omp task
                    {
                        for (int k = k_block; k < k_end; ++k) {
                            for (int i = i_block; i < i_end; ++i) {
                                if (dist[i][k] == INF) continue;
                                for (int j = k_block; j < k_end; ++j) {
                                    if (dist[k][j] == INF) continue;
                                    int new_dist = dist[i][k] + dist[k][j];
                                    if (new_dist < dist[i][j]) {
                                        dist[i][j] = new_dist;
                                    }
                                }
                            }
                        }
                    }
                    
                    #pragma omp task
                    {
                        for (int k = k_block; k < k_end; ++k) {
                            for (int j = i_block; j < i_end; ++j) {
                                if (dist[k][j] == INF) continue;
                                for (int i = k_block; i < k_end; ++i) {
                                    if (dist[i][k] == INF) continue;
                                    int new_dist = dist[i][k] + dist[k][j];
                                    if (new_dist < dist[i][j]) {
                                        dist[i][j] = new_dist;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i_block = 0; i_block < n; i_block += block_size) {
            for (int j_block = 0; j_block < n; j_block += block_size) {
                if (i_block == k_block || j_block == k_block) continue;
                
                int i_end = min(i_block + block_size, n);
                int j_end = min(j_block + block_size, n);
                
                for (int k = k_block; k < k_end; ++k) {
                    for (int i = i_block; i < i_end; ++i) {
                        if (dist[i][k] == INF) continue;
                        for (int j = j_block; j < j_end; ++j) {
                            if (dist[k][j] == INF) continue;
                            int new_dist = dist[i][k] + dist[k][j];
                            if (new_dist < dist[i][j]) {
                                dist[i][j] = new_dist;
                            }
                        }
                    }
                }
            }
        }
    }
}

double measure_time_seq(void (*sort_func)(vector<vector<int>>&), vector<vector<int>> dist) {
    auto start = high_resolution_clock::now();
    sort_func(dist);
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

double measure_time_par(void (*sort_func)(vector<vector<int>>&, int), 
                       vector<vector<int>> dist, int num_threads) {
    auto start = high_resolution_clock::now();
    sort_func(dist, num_threads);
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

vector<vector<int>> generate_dense_graph(int n) {
    vector<vector<int>> dist(n, vector<int>(n, INF));
    for (int i = 0; i < n; ++i) {
        dist[i][i] = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                dist[i][j] = rand() % 100 + 1;
            }
        }
    }
    return dist;
}

vector<vector<int>> generate_sparse_graph(int n) {
    vector<vector<int>> dist(n, vector<int>(n, INF));
    for (int i = 0; i < n; ++i) {
        dist[i][i] = 0;
        int edges = max(1, n / 10);
        for (int e = 0; e < edges; ++e) {
            int j = rand() % n;
            if (i != j) {
                dist[i][j] = rand() % 100 + 1;
            }
        }
    }
    return dist;
}

vector<vector<int>> generate_complete_graph(int n) {
    vector<vector<int>> dist(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                dist[i][j] = rand() % 100 + 1;
            }
        }
    }
    return dist;
}

bool verify_results(const vector<vector<int>>& result1, const vector<vector<int>>& result2) {
    int n = result1.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (result1[i][j] != result2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

void run_benchmarks() {
    vector<int> sizes = {100, 200, 400, 800};
    vector<int> threads = {2, 3, 4};
    vector<pair<string, function<vector<vector<int>>(int)>>> graph_generators = {
        {"Плотный граф (50% рёбер)", generate_dense_graph},
        {"Разреженный граф (10% рёбер)", generate_sparse_graph},
        {"Полный граф (100% рёбер)", generate_complete_graph}
    };
    
    cout << "\n╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                        АЛГОРИТМ ФЛОЙДА-УОРШЕЛЛА (БЛОЧНАЯ ВЕРСИЯ)                           ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n";
    
    for (const auto& [graph_name, generator] : graph_generators) {
        cout << "\n┌───────────────────────────────────────────────────────────────────────────────────────────────┐\n";
        cout << "│ " << graph_name << string(110 - graph_name.length(), ' ') << "│\n";
        cout << "├────────┬──────────┬──────────────┬──────────────┬───────────────┬─────────────────────────────┤\n";
        cout << "│ Размер │ Потоков  │ Послед.(ms)  │ Паралл.(ms)  │  Ускорение    │      Эффективность (%)      │\n";
        cout << "├────────┼──────────┼──────────────┼──────────────┼───────────────┼─────────────────────────────┤\n";
        
        for (int n : sizes) {
            auto graph = generator(n);
            
            double time_seq = measure_time_seq(floyd_warshall_seq, graph);
            
            for (int t : threads) {
                auto graph_copy = graph;
                double time_par = measure_time_par(floyd_warshall_par_blocked, graph_copy, t);
                double speedup = time_seq / time_par;
                double efficiency = (speedup / t) * 100.0;
                
                auto graph_seq = graph;
                auto graph_par = graph;
                floyd_warshall_seq(graph_seq);
                floyd_warshall_par_blocked(graph_par, t);
                bool correct = verify_results(graph_seq, graph_par);
                
                cout << "│ " << setw(6) << n << " │ " << setw(8) << t << " │ " 
                     << setw(12) << fixed << setprecision(2) << time_seq << " │ "
                     << setw(12) << time_par << " │ "
                     << setw(12) << setprecision(2) << speedup << "x │ "
                     << setw(26) << setprecision(1) << efficiency << "% │\n";
            }
            
            if (n != sizes.back()) {
                cout << "├────────┼──────────┼──────────────┼──────────────┼───────────────┼─────────────────────────────┤\n";
            }
        }
        cout << "└────────┴──────────┴──────────────┴──────────────┴───────────────┴─────────────────────────────┘\n";
        
        cout << "\nАНАЛИЗ ДЛЯ " << graph_name << ":\n";
        cout << "─────────────────────────────────────────────────────────────────\n";
        
        for (int n : sizes) {
            auto graph = generator(n);
            double time_seq = measure_time_seq(floyd_warshall_seq, graph);
            double time_par_4 = measure_time_par(floyd_warshall_par_blocked, graph, 4);
            double speedup_4 = time_seq / time_par_4;
            
            cout << "  • Размер n=" << n << ": ускорение на 4 ядрах = " 
                 << fixed << setprecision(2) << speedup_4 << "x";
            
            if (speedup_4 >= 3.5) {
                cout << " (отличная масштабируемость)";
            } else if (speedup_4 >= 2.5) {
                cout << " (хорошая масштабируемость)";
            } else {
                cout << " (умеренная масштабируемость)";
            }
            cout << "\n";
        }
    }
    
    cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                    СВОДНАЯ ТАБЛИЦА ЭФФЕКТИВНОСТИ                   ║\n";
    cout << "╠═══════════╦══════════════╦══════════════╦══════════════════════════╣\n";
    cout << "║  Размер   ║  2 потока    ║   3 потока   ║       4 потока           ║\n";
    cout << "╠═══════════╬══════════════╬══════════════╬══════════════════════════╣\n";
    
    for (int n : sizes) {
        cout << "║ " << setw(9) << n << " ║";
        
        for (int t : threads) {
            double avg_efficiency = 0.0;
            int count = 0;
            
            for (const auto& [name, generator] : graph_generators) {
                auto graph = generator(n);
                double time_seq = measure_time_seq(floyd_warshall_seq, graph);
                double time_par = measure_time_par(floyd_warshall_par_blocked, graph, t);
                double speedup = time_seq / time_par;
                avg_efficiency += (speedup / t) * 100.0;
                count++;
            }
            
            avg_efficiency /= count;
            cout << " " << setw(10) << fixed << setprecision(1) << avg_efficiency << "%  ║";
        }
        cout << "\n";
    }
    
    cout << "╚═══════════╩══════════════╩══════════════╩══════════════════════════╝\n";
}

int main() {
    srand(time(nullptr));
    
    cout << "Инициализация OpenMP...\n";
    cout << "Максимальное количество потоков: " << omp_get_max_threads() << "\n";
    cout << "Размер блока для блочного алгоритма: " << BLOCK_SIZE << "\n";
    
    run_benchmarks();
    
    cout << "\nВыполнение завершено успешно!\n";
    return 0;
}