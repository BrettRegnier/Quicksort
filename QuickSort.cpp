#include <iostream>
#include <vector>
#include <stack>
#include <random>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <pthread.h>

#define INSERT_THRESH 39

template <typename T>
void SerialInsertionSort(std::vector<T> &vec, int l, int r);

double StartSerialQuickSort1_5(int iter, int size);
template <typename T>
void SerialQuickSort1_5(std::vector<T> &vec, int l, int r);

double StartSerialQuickSort1_6(int iter, int size);
template <typename T>
void SerialQuickSort1_6(std::vector<T> &vec, int l, int r);

double StartStackQuickSort1_0(int iter, int size, int threads);
template <typename T>
void StackQuickSort1_0(std::vector<T> &vec, int l, int r, std::stack<std::pair<int, int>> &stack, int &busythreads, const int threads);

double StartStackQuickSort2_0(int iter, int size, int threads);
template <typename T>
void StackQuickSort2_0(std::vector<T> &vec, int l, int r, std::stack<std::pair<int, int>> &stack, int &busythreads, const int threads);

double StartNestedOMPSort2_0(int iter, int size, int threads);
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

    // t = StartSerialQuickSort1_5(3, s);
    // std::cout << "Elapsed Serial Quicksort v1.5 = " << t << std::endl;

    // t = StartSerialQuickSort1_6(3, s);
    // std::cout << "Elapsed Serial Quicksort v1.6 = " << t << std::endl;

    t = StartStackQuickSort1_0(3, s, 4);
    std::cout << "Elapsed Stack Quicksort v1.0 = " << t << std::endl;

    t = StartStackQuickSort2_0(3, s, 4);
    std::cout << "Elapsed Stack Quicksort v2.0 = " << t << std::endl;

    // t = StartNestedOMPSort1_0(3, s, 4);
    // std::cout << "Elapsed nested omp sort 1.0 = " << t << std::endl;

    // t = StartTaskQueueSort1_0(3, s, 4);
    // std::cout << "Elapsed nested omp sort 1.0 = " << t << std::endl;
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

double StartSerialQuickSort1_5(int iter, int size)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    std::vector<int> vec;
    for (int i = 0; i < iter; i++)
    {
        vec.clear();
        for (int i = 0; i < size; i++)
            vec.push_back(rand());

        start = omp_get_wtime();

        SerialQuickSort1_5(vec, 0, vec.size() - 1);

        stop = omp_get_wtime();
        elapsed += stop - start;
    }

    Validate(vec);

    return elapsed / iter;
}

template <typename T>
void SerialQuickSort1_5(std::vector<T> &vec, int l, int r)
{
    T pivot;
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
        std::swap(vec[i], vec[j]); // this is slower than just using a tmp, this will not be used outside of this function.
    }
    std::swap(vec[i], vec[r]); // this is slower than just using a tmp, this will not be used outside of this function.

    SerialQuickSort1_5(vec, l, i - 1);
    SerialQuickSort1_5(vec, i + 1, r);
}

double StartSerialQuickSort1_6(int iter, int size)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    std::vector<int> vec;
    for (int i = 0; i < iter; i++)
    {
        vec.clear();
        for (int i = 0; i < size; i++)
            vec.push_back(rand());

        start = omp_get_wtime();

        SerialQuickSort1_6(vec, 0, vec.size() - 1);

        stop = omp_get_wtime();
        elapsed += stop - start;
    }

    Validate(vec);

    return elapsed / iter;
}

template <typename T>
void SerialQuickSort1_6(std::vector<T> &vec, int l, int r)
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
        // this is the faster version
        tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
    // this is the faster verison
    tmp = vec[i];
    vec[i] = vec[r];
    vec[r] = tmp;

    SerialQuickSort1_6(vec, l, i - 1);
    SerialQuickSort1_6(vec, i + 1, r);
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
            l = r; // set area before l as sorted
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

double StartStackQuickSort2_0(int iter, int size, int threads)
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
                StackQuickSort2_0(vec, 0, vec.size() - 1, stack, busythreads, threads);
            else
                StackQuickSort2_0(vec, 0, 0, stack, busythreads, threads);
        }

        stop = omp_get_wtime();
        elapsed += (stop - start);
    }

    Validate(vec);

    return elapsed / iter;
}

template <typename T>
void StackQuickSort2_0(std::vector<T> &vec, int l, int r, std::stack<std::pair<int, int>> &stack, int &busythreads, const int threads)
{
    T pivot;
    T tmp;
    int i, j;
    bool idle = true;
    std::pair<int, int> bound;

    std::stack<std::pair<int, int>> localStack;

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
            if (!localStack.empty())
            {
                bound = localStack.top();
                localStack.pop();
                l = bound.first;
                r = bound.second;
            }
            else
            {
// Needs to be critical because multiple threads will want access to the stack
#pragma omp critical
                {
                    // if there is stuff on the stack
                    if (!stack.empty())
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
                        if (!idle)
                            --busythreads;
                        idle = true;
                    }
                }
                // break out if the all threads are done looking for work in the stack
                if (busythreads == 0)
                    return;
            } // end else
        }     // end while

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

            if (stack.size() < 16)
#pragma omp critical
                stack.push(bound);
            else
                localStack.push(bound);
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
    for (int i = 0; i < iter; i++)
    {
        vec.clear();
        for (int i = 0; i < size; i++)
            vec.push_back(rand());

        int busythreads = 1;

        start = omp_get_wtime();

        NestedOMPSort1_0(vec, 0, vec.size() - 1, busythreads, threads);

        stop = omp_get_wtime();
        elapsed += (stop - start);
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
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    std::vector<int> vec;
    for (int i = 0; i < iter; i++)
    {
        vec.clear();
        for (int i = 0; i < size; i++)
            vec.push_back(rand());

        start = omp_get_wtime();

#pragma omp parallel num_threads(threads) shared(vec)
#pragma omp single
        TaskQueueSort1_0(vec, 0, vec.size() - 1);

        stop = omp_get_wtime();
        elapsed += (stop - start);
    }

    Validate(vec);

    return elapsed / iter;
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

#pragma omp task shared(vec)
    {
        TaskQueueSort1_0(vec, l, i - 1);
    }
#pragma omp task shared(vec)
    {
        TaskQueueSort1_0(vec, i + 1, r);
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