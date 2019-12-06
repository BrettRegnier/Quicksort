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
template <typename T>
void SerialInsertionSort(T *vec, int l, int r);

double StartSerialQuickSort1_5(int iter, int size);
template <typename T>
void SerialQuickSort1_5(std::vector<T> &vec, int l, int r);

double StartSerialQuickSort1_6(int iter, int size);
template <typename T>
void SerialQuickSort1_6(T *vec, int l, int r);

double StartStackQuickSort1_0(int iter, int size, int threads);
template <typename T>
void StackQuickSort1_0(T *vec, int low, int high, std::stack<int> &stack, int &busythreads, const int threads);

double StartStackQuickSort2_0(int iter, int size, int threads);
template <typename T>
void StackQuickSort2_0(T *vec, int l, int r, std::stack<int> &stack, int &busythreads, const int threads);

double StartNestedOMPSort1_0(int iter, int size, int threads);
template <typename T>
void NestedOMPSort1_0(T *vec, int l, int r, int &busythreads, int &threads);

double StartTaskQueueSort1_0(int iter, int size, int threads);
template <typename T>
void TaskQueueSort1_0(T *vec, int l, int r);

double StartPThreadsSort(int iter, int size, int threads);
void *PThreadsRunner(void *param);
template <typename T>
void PThreadsSort(T *vec, int l, int r, int &activeThreads, int maxThreads);

template <typename T>
void FillArray(T *vec, int s);
template <typename T>
bool Validate(std::vector<T> &vec);

// The validate function is only ran at the end of a function call because I only really need to check if its correct at the end after each time its been
// attempted to be sorted.
template <typename T>
bool Validate(T *vec, int s);
void Output(double time, int s, int threads, char const name[100]);

template <typename T>
struct pThreadObj
{
    T* vec;
    int l;
    int r;
    int &activeThreads;
    int maxThreads;
};

int main()
{
    std::cout << "threads available = " << omp_get_max_threads() << std::endl;

    double t;
    int s = 100000000;
    // s = 1000000;
    int i;

    t = StartSerialQuickSort1_5(3, s);
    Output(t, s, 1, "-----------Serial Quicksort v1.5 Metrics-----------");

    t = StartSerialQuickSort1_6(3, s);
    Output(t, s, 1, "-----------Serial Quicksort v1.6 Metrics-----------");

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartStackQuickSort1_0(3, s, i);
        Output(t, s, i, "-----------Stack Quicksort v1.0 Metrics-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartStackQuickSort2_0(3, s, i);
        Output(t, s, i, "-----------Stack Quicksort v2.0 Metrics-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartNestedOMPSort1_0(3, s, i);
        Output(t, s, i,  "-----------Nested OMP Quicksort v1.0 Metrics-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartTaskQueueSort1_0(3, s, i);
        Output(t, s, i,  "-----------Task Queue Quicksort v1.0-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartPThreadsSort(3, s, i);
        Output(t, s, i,  "-----------PThreads Quicksort v1.0-----------");
    }
}

template <typename T>
void SerialInsertionSort(std::vector<T> &vec, int low, int high)
{
    int i = low;
    int j;
    T tmp;
    while (i <= high)
    {
        j = i;
        while (j > 0 && vec[j - 1] > vec[j])
        {
            tmp = vec[j];
            vec[j] = vec[j - 1];
            vec[j - 1] = tmp;
            j = j - 1;
        }
        i = i + 1;
    }
}
template <typename T>
void SerialInsertionSort(T *vec, int low, int high)
{
    int i = low;
    int j;
    T tmp;
    while (i <= high)
    {
        j = i;
        while (j > 0 && vec[j - 1] > vec[j])
        {
            tmp = vec[j];
            vec[j] = vec[j - 1];
            vec[j - 1] = tmp;
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
void SerialQuickSort1_5(std::vector<T> &vec, int low, int high)
{
    T pivot;
    int i, j;

    if (high - low < INSERT_THRESH)
    {
        SerialInsertionSort(vec, low, high);
        return;
    }

    pivot = vec[high];
    i = low - 1;
    j = high;

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
    std::swap(vec[i], vec[high]); // this is slower than just using a tmp, this will not be used outside of this function.

    SerialQuickSort1_5(vec, low, i - 1);
    SerialQuickSort1_5(vec, i + 1, high);
}

double StartSerialQuickSort1_6(int iter, int size)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    int *vec;
    for (int i = 0; i < iter; i++)
    {
        // make the array and fil lit
        vec = (int *)malloc(size * sizeof(int));
        FillArray(vec, size);

        start = omp_get_wtime();

        SerialQuickSort1_6(vec, 0, size - 1);

        stop = omp_get_wtime();
        elapsed += stop - start;

        // free the array so its back to an empty arr, and then prepare it again for the next loop.
        if (i + 1 < iter)
            free(vec);
    }

    Validate(vec, size);
    free(vec);

    return elapsed / iter;
}

template <typename T>
void SerialQuickSort1_6(T *vec, int low, int high)
{
    T pivot;
    T tmp;
    int i, j;

    if (high - low < INSERT_THRESH)
    {
        SerialInsertionSort(vec, low, high);
        return;
    }

    pivot = vec[high];
    i = low - 1;
    j = high;

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
    vec[i] = vec[high];
    vec[high] = tmp;

    SerialQuickSort1_6(vec, low, i - 1);
    SerialQuickSort1_6(vec, i + 1, high);
}

double StartStackQuickSort1_0(int iter, int size, int threads)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    int *vec;
    for (int i = 0; i < iter; i++)
    {
        // make the array and fill it
        vec = (int *)malloc(size * sizeof(int));
        FillArray(vec, size);

        int busythreads = 1;
        std::stack<int> stack;

        start = omp_get_wtime();

#pragma omp parallel num_threads(threads) shared(vec, stack, threads, busythreads)
        {
            if (omp_get_thread_num() == 0)
                StackQuickSort1_0(vec, 0, size - 1, stack, busythreads, threads);
            else
                StackQuickSort1_0(vec, 0, 0, stack, busythreads, threads);
        }

        stop = omp_get_wtime();
        elapsed += (stop - start);

        // Free the array so its back to an empty arr, and then prepare it again for the next loop unless its the end
        if (i + 1 < iter)
            free(vec);
    }

    Validate(vec, size);
    free(vec);

    return elapsed / iter;
}

template <typename T>
void StackQuickSort1_0(T *vec, int low, int high, std::stack<int> &stack, int &busythreads, const int threads)
{
    T pivot;
    T tmp;
    int i, j;
    bool idle = true;

    if (low != high)
        idle = false;

    // begin processing loop for a thread(s)
    do
    {
        if (high - low < INSERT_THRESH)
        {
            SerialInsertionSort(vec, low, high);
            low = high; // set area before low as sorted and move on!
        }

        while (low >= high)
        {
#pragma omp critical
            {
                // if there is stuff on the stack
                if (stack.empty() == false)
                {
                    if (idle)
                        ++busythreads;
                    idle = false;

                    // Pop low and high values off the stack and process them.
                    low = stack.top();
                    stack.pop();
                    high = stack.top();
                    stack.pop();
                }
                else
                {
                    if (idle == false)
                        --busythreads;
                    idle = true;
                }
            }
            // break out if the all threads are done and no work to do on the stack
            if (busythreads == 0)
                return;
        }

        // Regular quicksort here, I could use a function call but that is slower.
        pivot = vec[high];
        i = low - 1;
        j = high;

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
        }
        tmp = vec[i];
        vec[i] = vec[high];
        vec[high] = tmp;

        // If there is a lot of work to do stil put stuff on the stack.
        if (i - 1 - low > INSERT_THRESH)
        {
// Ensure only one thread can push on the stack at a time.
// Operation is so quick this almost isn't needed.
#pragma omp critical
            {
                stack.push(i - 1);
                stack.push(low);
            }
        }
        else
            SerialInsertionSort(vec, low, i - 1);

        low = i + 1;
    } while (true);
}

double StartStackQuickSort2_0(int iter, int size, int threads)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    int *vec;
    for (int i = 0; i < iter; i++)
    {
        // make the array and fill it
        vec = (int *)malloc(size * sizeof(int));
        FillArray(vec, size);

        int busythreads = 1;
        std::stack<int> stack;

        start = omp_get_wtime();

#pragma omp parallel num_threads(threads) shared(vec, stack, threads, busythreads)
        {
            if (omp_get_thread_num() == 0)
                StackQuickSort2_0(vec, 0, size - 1, stack, busythreads, threads);
            else
                StackQuickSort2_0(vec, 0, 0, stack, busythreads, threads);
        }

        stop = omp_get_wtime();
        elapsed += (stop - start);

        // Free the array so its back to an empty arr, and then prepare it again for the next loop unless its the end
        if (i + 1 < iter)
            free(vec);
    }

    Validate(vec, size);
    free(vec);

    return elapsed / iter;
}

template <typename T>
void StackQuickSort2_0(T *vec, int low, int high, std::stack<int> &stack, int &busythreads, const int threads)
{
    T pivot;
    T tmp;
    int i, j;
    bool idle = true;

    std::stack<int> localStack;

    if (low != high)
        idle = false;

    while (true)
    {
        if (high - low < INSERT_THRESH)
        {
            SerialInsertionSort(vec, low, high);
            low = high;
        }

        while (low >= high)
        {
            if (!localStack.empty())
            {
                // pop the low and high indices off the local stack
                low = localStack.top();
                localStack.pop();
                high = localStack.top();
                localStack.pop();
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

                        // pop the low and the high indices off the global stack.
                        low = stack.top();
                        stack.pop();
                        high = stack.top();
                        stack.pop();
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

        pivot = vec[high];
        i = low - 1;
        j = high;

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
        }
        tmp = vec[i];
        vec[i] = vec[high];
        vec[high] = tmp;

        if (i - 1 - low > INSERT_THRESH)
        {
            if (stack.size() < 16)
            {
// only want 1 core pushing onto the stack at a time, incase write problems.
#pragma omp critical
                {
                    // push the high on first, because low is expected to be on top first.
                    stack.push(i - 1);
                    stack.push(low);
                }
            }
            else
            {
                // push the high on first, because low is expected to be on top first.
                localStack.push(i - 1);
                localStack.push(low);
            }
        }
        else
            SerialInsertionSort(vec, low, i - 1);

        low = i + 1;
    }
}

double StartNestedOMPSort1_0(int iter, int size, int threads)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    int *vec;
    for (int i = 0; i < iter; i++)
    {
        // make the array and fill it
        vec = (int *)malloc(size * sizeof(int));
        FillArray(vec, size);

        int busythreads = 1;

        start = omp_get_wtime();

        NestedOMPSort1_0(vec, 0, size - 1, busythreads, threads);

        stop = omp_get_wtime();
        elapsed += (stop - start);

        if (i + 1 < iter)
            free(vec);
    }

    Validate(vec, size);
    free(vec);

    return elapsed / iter;
}

template <typename T>
void NestedOMPSort1_0(T *vec, int l, int r, int &busythreads, int &threads)
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
        busythreads += 2;

#pragma omp parallel num_threads(threads) shared(vec, threads, busythreads, i, l, r)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    NestedOMPSort1_0(vec, l, i - 1, busythreads, threads);

                    // this occurs because it will happen after the thread has come about out of the recursive step
                    busythreads--;
                }
#pragma omp section
                {
                    NestedOMPSort1_0(vec, i + 1, r, busythreads, threads);

                    // this occurs because it will happen after the thread has come about out of the recursive step
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
    int *vec;
    for (int i = 0; i < iter; i++)
    {
        // make the array and fill it
        vec = (int *)malloc(size * sizeof(int));
        FillArray(vec, size);

        start = omp_get_wtime();

#pragma omp parallel num_threads(threads) shared(vec)
#pragma omp single
        TaskQueueSort1_0(vec, 0, size - 1);

        stop = omp_get_wtime();
        elapsed += (stop - start);

        // Free the array so its back to an empty arr, and then prepare it again for the next loop unless its the end
        if (i + 1 < iter)
            free(vec);
    }

    Validate(vec, size);
    free(vec);

    return elapsed / iter;
}

template <typename T>
void TaskQueueSort1_0(T *vec, int l, int r)
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

double StartPThreadsSort(int iter, int size, int threads)
{
    double elapsed = 0.0f;
    double start = 0.0f;
    double stop = 0.0f;
    int *vec;
    for (int i = 0; i < iter; i++)
    {
        // make the array and fill it
        vec = (int *)malloc(size * sizeof(int));
        FillArray(vec, size);
        
        int activeThreads = 1;

        start = omp_get_wtime();

        PThreadsSort(vec, 0, size - 1, activeThreads, threads);

        stop = omp_get_wtime();
        elapsed += (stop - start);

        // Free the array so its back to an empty arr, and then prepare it again for the next loop unless its the end
        if (i + 1 < iter)
            free(vec);
    }

    Validate(vec, size);
    free(vec);

    return elapsed / iter;
}

void *PThreadsRunner(void *param)
{
    struct pThreadObj<int> p = *(static_cast<pThreadObj<int> *>(param));
    p.activeThreads++;

    PThreadsSort(p.vec, p.l, p.r, p.activeThreads, p.maxThreads);
    return NULL;
}

template <typename T>
void PThreadsSort(T *vec, int l, int r, int &activeThreads, int const maxThreads)
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
    }
    tmp = vec[i];
    vec[i] = vec[r];
    vec[r] = tmp;

    if (activeThreads < maxThreads)
    {
        pthread_t thread;
        struct pThreadObj<int> p =
        {
            vec, l, i - 1, activeThreads, maxThreads
        };

        // create a new thread and process it.
        pthread_create(&thread, NULL, PThreadsRunner, &p);
        PThreadsSort(vec, i + 1, r, activeThreads, maxThreads);

        pthread_join(thread, NULL);
        activeThreads--;
    }
    else
    {
        // all threads are busy, do a serial sort instead
        PThreadsSort(vec, l, i - 1, activeThreads, maxThreads);
        PThreadsSort(vec, i + 1, r, activeThreads, maxThreads);
    }
}

template <typename T>
void FillArray(T *vec, int s)
{
    srand(777);
    for (int i = 0; i < s; i++)
        vec[i] = rand();
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

template <typename T>
bool Validate(T *vec, int s)
{
    bool isSorted = true;
    for (int i = 1; i < s; i++)
    {
        if (vec[i - 1] > vec[i])
            isSorted = false;
    }

    std::cout << "\nSorted = " << isSorted << std::endl;
    return isSorted;
}

void Output(double time, int s, int threads, char const name[100])
{

    std::cout << name << std::endl;
    std::cout << "Threads = " << threads << std::endl;
    std::cout << "Array Size = " << s << std::endl;
    std::cout << "Time Elapsed = " << time << std::endl;
}