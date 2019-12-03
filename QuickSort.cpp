#include <iostream>
#include <vector>
#include <stack>
#include <random>
#include <chrono>
#include <omp.h>
#include <stdio.h>

#define INSERT_THRESH 39

template <typename T>
void SerialInsertionSort(std::vector<T> &vec, int l, int r);

double StartSerialQuickSort(int iter, int size);
template <typename T>
void SerialQuickSort(std::vector<T> &vec, int l, int r);

double StartStackQuickSort1_0(int iter, int size, int threads);
template <typename T>
void StackQuickSort1_0(std::vector<T> &vec, int l, int r, std::stack<std::pair<int, int>> &stack, int &busythreads, const int threads);

double StartNestedOMPSort1_0(int iter, int size, int threads);
template <typename T>
void NestedOMPSort1_0(std::vector<T> &vec, int l, int r, int &busythreads, int &threads);

double StartTaskQueueSort1_0(int iter, int size, int threads);
template <typename T>
void TaskQueueSort1_0(std::vector<T> &vec, int l, int r);

template <typename T>
bool Validate(std::vector<T> &vec);
template <typename T>
void PrintResults(std::vector<T> &vec);

int main()
{
    std::cout << "threads available = " << omp_get_max_threads() << std::endl;

    double t;
    int s = 10000000;

    // t = StartSerialQuickSort(1, s);
    // std::cout << "Elapsed serial sort = " << t << std::endl;

    // t = StartStackQuickSort1_0(1, s, 4);
    // std::cout << "Elapsed stack sort 1.0 = " << t << std::endl;

    t = StartNestedOMPSort1_0(1, s, 4);
    std::cout << "Elapsed nested omp sort 1.0 = " << t << std::endl;
}

template <typename T>
void SerialInsertionSort(std::vector<T> &vec, int l, int r)
{
    int i = l;
    int j;
    while (i <= r)
    {
        j = i;
        while (j > 0 && vec[j - 1] > vec[j])
        {
            std::swap(vec[j], vec[j - 1]);
            j = j - 1;
        }
        i = i + 1;
    }
}

double StartSerialQuickSort(int iter, int size)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    std::vector<int> vec;
    for (int i = 0; i < iter; i++)
    {
        vec.clear();
        int n = size;
        for (int i = 0; i < n; i++)
            vec.push_back(rand());

        start = omp_get_wtime();

        SerialQuickSort(vec, 0, vec.size() - 1);

        stop = omp_get_wtime();
        elapsed += stop - start;
    }

    Validate(vec);

    return elapsed / iter;
}

template <typename T>
void SerialQuickSort(std::vector<T> &vec, int l, int r)
{
    T pivot;
    T tmp;
    int i, j;

    if (r - l < INSERT_THRESH)
    {
        SerialInsertionSort(vec, l, r);
        return;
    }

    pivot = vec[r];
    i = l - 1;
    j = r;

    while (true)
    {
        while (vec[++i] < pivot)
            ;
        while (vec[--j] > pivot)
            ;
        if (i >= j)
            break;
        tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
        // std::swap(vec[i], vec[j]); // this is slower than just using a tmp.
    }
    tmp = vec[i];
    vec[i] = vec[r];
    vec[r] = tmp;
    // std::swap(vec[i], vec[r]); // this is slower than just using a tmp

    SerialQuickSort(vec, l, i - 1);
    SerialQuickSort(vec, i + 1, r);
}

double StartStackQuickSort1_0(int iter, int size, int threads)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    std::vector<int> vec;
    for (int i = 0; i < iter; i++)
    {
        vec.clear();
        int busythreads = 1;
        std::stack<std::pair<int, int>> stack;

        for (int i = 0; i < size; i++)
            vec.push_back(rand());

        start = omp_get_wtime();

#pragma omp parallel num_threads(threads) shared(vec, stack, threads, busythreads)
        {
            if (omp_get_thread_num() == 0)
                StackQuickSort1_0(vec, 0, vec.size() - 1, stack, busythreads, threads);
            else
                StackQuickSort1_0(vec, 0, 0, stack, busythreads, threads);
        }

        stop = omp_get_wtime();
        elapsed += (stop - start);
    }

    Validate(vec);

    return elapsed / iter;
}

template <typename T>
void StackQuickSort1_0(std::vector<T> &vec, int l, int r, std::stack<std::pair<int, int>> &stack, int &busythreads, const int threads)
{
    T pivot;
    T tmp;
    int i, j;
    bool idle = true;
    std::pair<int, int> bound;

    if (l != r)
        idle = false;

    while (true)
    {
        if (r - l < INSERT_THRESH)
        {
            SerialInsertionSort(vec, l, r);
            l = r;
        }

        while (l >= r)
        {
#pragma omp critical
            {
                // if there is stuff on the stack
                if (stack.empty() == false)
                {
                    if (idle)
                        ++busythreads;
                    idle = false;

                    bound = stack.top();
                    stack.pop();
                    l = bound.first;
                    r = bound.second;
                }
                else
                {
                    if (idle == false)
                        --busythreads;
                    idle = true;
                }
            }
            // break out if the all threads are done
            if (busythreads == 0)
                return;
        }

        pivot = vec[r];
        i = l - 1;
        j = r;

        while (true)
        {
            // std::cout << "here1" << std::endl;
            while (vec[++i] < pivot)
                ;
            while (vec[--j] > pivot)
                ;
            if (i >= j)
                break;
            tmp = vec[i];
            vec[i] = vec[j];
            vec[j] = tmp;
        }
        tmp = vec[i];
        vec[i] = vec[r];
        vec[r] = tmp;

        if (i - 1 - l > INSERT_THRESH)
        {
            bound = std::make_pair(l, i - 1);

#pragma omp critical
            {
                stack.push(bound);
            }
        }
        else
            SerialInsertionSort(vec, l, i - 1);

        l = i + 1;
    }
}

double StartNestedOMPSort1_0(int iter, int size, int threads)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    std::vector<int> vec;
    std::vector<double> timing;
    for (int j = 10; j < 200; j++)
    {
            vec.clear();
            int n = size;
            for (int i = 0; i < n; i++)
                vec.push_back(rand());

            int busythreads = 1;

            start = omp_get_wtime();

            NestedOMPSort1_0(vec, 0, vec.size() - 1, busythreads, threads);

            stop = omp_get_wtime();
            elapsed += (stop - start);
        timing.push_back(elapsed / iter);
        elapsed = 0.0f;
    }

    Validate(vec);

    return elapsed / iter;
}

template <typename T>
void NestedOMPSort1_0(std::vector<T> &vec, int l, int r, int &busythreads, int &threads)
{
    T pivot;
    T tmp;
    int i, j;

    if (r - l < INSERT_THRESH)
    {
        SerialInsertionSort(vec, l, r);
        return;
    }

    pivot = vec[r];
    i = l - 1;
    j = r;

    while (true)
    {
        // std::cout << "here1" << std::endl;
        while (vec[++i] < pivot)
            ;
        while (vec[--j] > pivot)
            ;
        if (i >= j)
            break;
        tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
    tmp = vec[i];
    vec[i] = vec[r];
    vec[r] = tmp;

    if (busythreads >= threads)
    {
        NestedOMPSort1_0(vec, l, i - 1, busythreads, threads);
        NestedOMPSort1_0(vec, i + 1, r, busythreads, threads);
    }
    else
    {
#pragma omp atomic update
        busythreads += 2;

#pragma omp parallel num_threads(threads) shared(vec, threads, busythreads, i, l, r)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    NestedOMPSort1_0(vec, l, i - 1, busythreads, threads);

                    // this occurs because it will happen after the thread has come about out of the recursive step
#pragma omp atomic
                    busythreads--;
                }
#pragma omp section
                {
                    NestedOMPSort1_0(vec, i + 1, r, busythreads, threads);

                    // this occurs because it will happen after the thread has come about out of the recursive step
#pragma omp atomic
                    busythreads--;
                }
            }
        }
    }
}

double StartTaskQueueSort1_0(int iter, int size, int threads)
{
}

template <typename T>
void TaskQueueSort1_0(std::vector<T> &vec, int l, int r)
{

    T pivot;
    T tmp;
    int i, j;

    if (r - l < INSERT_THRESH)
    {
        SerialInsertionSort(vec, l, r);
        return;
    }

    pivot = vec[r];
    i = l - 1;
    j = r;

    while (true)
    {
        // std::cout << "here1" << std::endl;
        while (vec[++i] < pivot)
            ;
        while (vec[--j] > pivot)
            ;
        if (i >= j)
            break;
        tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
    tmp = vec[i];
    vec[i] = vec[r];
    vec[r] = tmp;

#pragma omp tasq
    {
    }
}

template <typename T>
bool Validate(std::vector<T> &vec)
{
    int leng = vec.size();
    bool isSorted = true;
    for (int i = 1; i < leng; i++)
    {
        if (vec[i - 1] > vec[i])
            isSorted = false;
    }

    std::cout << "\nSorted = " << isSorted << std::endl;
    return isSorted;
}