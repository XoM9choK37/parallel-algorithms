#include <iostream>
#include <omp.h>
#include <ctime>
#include <fstream>
using namespace std;
int main()
{
    srand(time(0));
    setlocale(LC_ALL, "RU");

    std::cout << "Hello World!\n";
#ifdef _OPENMP
    std::cout << "Yes";
#endif
#ifdef _OPENMP
    {
#pragma omp parallel
        {
            std::cout << "я поток номер " << omp_get_thread_num() << endl;
        }
    }
#endif
    omp_set_nested(1);

    int ALL_IN = 9;
    int n;
    /*std::cout << "Укажите размерность матриц: n " << endl << "n=";
    std::cin >> n;
    std::cout << endl;*/
    int min = 1;
    int max = 20;
    double start_time;
    double end_time;
    int count_consts_num_threads = 3;
    int* consts_num_threads = new int[count_consts_num_threads]{ 2, 3, 4 };
    int* consts_n = new int[ALL_IN]{ 10, 15, 30, 60, 120, 250, 500, 1000, 2000 };
    ofstream fout;
    fout.open("F:\\Учоба\\3 курс 6 семестр\\Параллельность\\file.txt");
    for (int zzz = 0; zzz < ALL_IN; zzz++)
    {
        n = consts_n[zzz];
        int** matrix1 = new int* [n];
        int** matrix2 = new int* [n];
        for (int i = 0; i < n; i++)
        {
            matrix1[i] = new int[n];
            matrix2[i] = new int[n];
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                // std::cin >> matrix1[i][j];
                matrix1[i][j] = rand() % (max - min + 1) + min;
            }
        }
        std::cout << "Элементы первой матрицы:" << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                // std::cin >> matrix2[i][j];
                matrix2[i][j] = rand() % (max - min + 1) + min;
            }
        }
        std::cout << "Элементы второй матрицы:" << endl;
        int** result_matrix1 = new int* [n];
        for (int i = 0; i < n; i++)
        {
            result_matrix1[i] = new int[n];

        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result_matrix1[i][j] = 0;
            }
        }
        // int i, j, g;
        // Последовательный
        start_time = omp_get_wtime();
        // int i, j, g;       
        for (int i = 0; i < n; i++)
        {                
            for (int j = 0; j < n; j++)
            {
                for (int g = 0; g < n; g++)
                {
                    result_matrix1[i][j] += matrix1[i][g] * matrix2[g][j];
                }                        
            }                
        }        
        end_time = omp_get_wtime();
        std::cout << std::fixed << "Последовательное перемножение матриц" << endl << "time: " << end_time - start_time << endl;
        fout << n << ";" << "Последовательный" << ";" << end_time - start_time << endl;


        for (int zz = 0; zz < count_consts_num_threads; zz++)
        {

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    result_matrix1[i][j] = 0;
                }
            }

            // PARALLEL
            start_time = omp_get_wtime();

            int i, j, g;
            #pragma omp parallel for collapse(3) private(i, j, g) shared(result_matrix1) num_threads(consts_num_threads[zz])
            for (i = 0; i < n; i++)
            {
                for (j = 0; j < n; j++)
                {
                    for (g = 0; g < n; g++)
                    {
                        result_matrix1[i][j] += matrix1[i][g] * matrix2[g][j];
                    }
                }
            }

            double end_time = omp_get_wtime();
            std::cout << std::fixed << "Параллельное перемножение матриц" << endl << "time: " << end_time - start_time << endl;
            fout << n << ";" << "Потоки" << consts_num_threads[zz] << ";" << end_time - start_time << endl;
        }
    }
    fout.close();
}
