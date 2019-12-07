#include <iostream>
#include <vector>
#include <stack>
#include <random>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <pthread.h>

// Found based on running experiments and finding the best threshold
// based on the fastest time after running multiple times to get this number
#define INSERT_THRESH 39

// Basic Serial Insertion Sort 2 versions.
template <typename T>
void SerialInsertionSort(std::vector<T> &vec, int l, int r);
template <typename T>
void SerialInsertionSort(T *vec, int l, int r);

double StartSerialQuickSort1_5(int iter, int size);
template <typename T>
void SerialQuickSort1_5(std::vector<T> &vec, int l, int r);

double StartSerialQuickSort2_6(int iter, int size);
template <typename T>
void SerialQuickSort2_6(T *vec, int low, int high);

double StartStackQuickSort3_0(int iter, int size, int threads);
template <typename T>
void StackQuickSort3_0(T *vec, int low, int high, std::stack<int> &stack, int &busythreads);

double StartStackQuickSort4_0(int iter, int size, int threads);
template <typename T>
void StackQuickSort4_0(T *vec, int low, int high, std::stack<int> &stack, int &busythreads);

double StartNestedOMPSort2_0(int iter, int size, int threads);
template <typename T>
void NestedOMPSort2_0(T *vec, int low, int high, int &busythreads, int &threads);

double StartTaskQueueQuickSort2_0(int iter, int size, int threads);
template <typename T>
void TaskQueueQuickSort2_0(T *vec, int l, int r);

double StartPThreadsQuickSort2_0(int iter, int size, int threads);
void *PThreadsRunner(void *param);
template <typename T>
void PThreadsQuickSort2_0(T *vec, int l, int r, int &activeThreads, int maxThreads);

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
    T *vec;
    int low;
    int high;
    int &activeThreads;
    int maxThreads;
};

int main()
{
    std::cout << "threads available = " << omp_get_max_threads() << std::endl;

    double t;
    int s = 100000000;
    int i;

    t = StartSerialQuickSort1_5(3, s);
    Output(t, s, 1, "-----------Serial Quicksort v1.5 Metrics-----------");

    t = StartSerialQuickSort2_6(3, s);
    Output(t, s, 1, "-----------Serial Quicksort v2.6 Metrics-----------");

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartStackQuickSort3_0(3, s, i);
        Output(t, s, i, "-----------Stack Quicksort v3.0 Metrics-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartStackQuickSort4_0(3, s, i);
        Output(t, s, i, "-----------Stack Quicksort v4.0 Metrics-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartNestedOMPSort2_0(3, s, i);
        Output(t, s, i, "-----------Nested OMP Quicksort v2.0 Metrics-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartTaskQueueQuickSort2_0(3, s, i);
        Output(t, s, i, "-----------Task Queue Quicksort v2.0 Metrics-----------");
    }

    for (i = 1; i <= 8; i *= 2)
    {
        t = StartPThreadsQuickSort2_0(3, s, i);
        Output(t, s, i, "-----------PThreads Quicksort v2.0 Metrics-----------");
    }
}

/*
    Serial Insertion Sort
    
    std::vector<T> &vec = Reference to a vector of template T type
    int low = the low index
    int high = the high index
*/
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

/*
    Serial Insertion Sort
    
    T *vec = Pointer to an array of template T type
    int low = the low index
    int high = the high index
*/
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

/*
    Starter for Serial Quicksort v1_5
    
    int iter = number of times to run the algorithm to get an average run time
    int size = siez of the array
*/
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

/*
    Serial Quicksort v1.5
    
    std::vector<T> = Reference to vector of template T type
    int low = the low index
    int high = the high index
*/
template <typename T>
void SerialQuickSort1_5(std::vector<T> &vec, int low, int high)
{
    T pivot;
    int i, j;

    // if there is not a lot of work left just do a serial insertion sort.
    if (high - low < INSERT_THRESH)
    {
        SerialInsertionSort(vec, low, high);
        return;
    }

    // value to pivot around
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

/*
    Starter for Serial Quicksort v2_6
    
    int iter = number of times to run the algorithm to get an average run time
    int size = siez of the array
*/
double StartSerialQuickSort2_6(int iter, int size)
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

        SerialQuickSort2_6(vec, 0, size - 1);

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

/*
    Serial Quicksort v1.5
    
    std::vector<T> vec = Reference to vector of template T type
    int low = the low index
    int high = the high index
*/
template <typename T>
void SerialQuickSort2_6(T *vec, int low, int high)
{
    T pivot;
    T tmp;
    int i, j;

    // if there is not a lot of work left just do a serial insertion sort.
    if (high - low < INSERT_THRESH)
    {
        SerialInsertionSort(vec, low, high);
        return;
    }

    // value to pivot around
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

    // Recursive call
    SerialQuickSort2_6(vec, low, i - 1);
    SerialQuickSort2_6(vec, i + 1, high);
}

/*
    Starter for Stack Quicksort v3_0
    
    int iter = number of times to run the algorithm to get an average run time
    int size = size of the array
    int threads = number of threads to use
    
    return double = elapsed average time
*/
double StartStackQuickSort3_0(int iter, int size, int threads)
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

        // current number of stacks that are pulling from the stack
        int busythreads = 1;
        // this is the global stack
        std::stack<int> stack;

        start = omp_get_wtime();

#pragma omp parallel num_threads(threads) shared(vec, stack, threads, busythreads)
        {
            if (omp_get_thread_num() == 0)
                // Need to have the master thread put stuff onto the stack first.
                StackQuickSort3_0(vec, 0, size - 1, stack, busythreads);
            else
                // The other threads will get their low and high from the stack.
                StackQuickSort3_0(vec, 0, 0, stack, busythreads);
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

/*
    Stack Quicksort v3.0
    
    T *vec = Pointer to an array of template T type
    int low = the low index
    int high = the high index
    std::stack<int> &stack = a shared stack that stores indices of the array
    int &busythreads = a shared counter that determines if the stack can be popped off of
*/
template <typename T>
void StackQuickSort3_0(T *vec, int low, int high, std::stack<int> &stack, int &busythreads)
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

/*
    Starter for Stack Quicksort v3_0
    
    int iter = number of times to run the algorithm to get an average run time
    int size = size of the array
    int threads = number of threads to use
    
    return double = elapsed average time
*/
double StartStackQuickSort4_0(int iter, int size, int threads)
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

#pragma omp parallel num_threads(threads) shared(vec, stack, busythreads)
        {
            if (omp_get_thread_num() == 0)
            // Need to have the master thread put stuff onto the stack first.
                StackQuickSort4_0(vec, 0, size - 1, stack, busythreads);
            else
                // The other threads will get their low and high from the stack.
                StackQuickSort4_0(vec, 0, 0, stack, busythreads);
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

/*
    Stack Quicksort v4.0
    
    T *vec = Pointer to an array of template T type
    int low = the low index
    int high = the high index
    std::stack<int> &stack = a shared stack that stores indices of the array
    int &busythreads = a shared counter that determines if the stack can be popped off of
    const int threads = number of threads that can be used.
*/
template <typename T>
void StackQuickSort4_0(T *vec, int low, int high, std::stack<int> &stack, int &busythreads)
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

        // Same as the normal serial quicksort v2.6
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
        // until here.

        // if there is still work to do that is greater than the threshold
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

/*
    Starter for Nested OMP Sort v2.0
    
    int iter = number of times to run the algorithm to get an average run time
    int size = size of the array
    int threads = number of threads to use
    
    return double = elapsed average time
*/
double StartNestedOMPSort2_0(int iter, int size, int threads)
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

        NestedOMPSort2_0(vec, 0, size - 1, busythreads, threads);

        stop = omp_get_wtime();
        elapsed += (stop - start);

        if (i + 1 < iter)
            free(vec);
    }

    Validate(vec, size);
    free(vec);

    return elapsed / iter;
}

/*
    Nested OpenMP Sort with sections
    
    T *vec = Pointer to an array of template T type
    int low = the low index
    int high = the high index
    std::stack<int> &stack = a shared stack that stores indices of the array
    int &busythreads = a shared counter that determines if the stack can be popped off of
    int &threads = current active threads
*/
template <typename T>
void NestedOMPSort2_0(T *vec, int low, int high, int &busythreads, int &threads)
{
    // Regular serial quicksort
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
        tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
    tmp = vec[i];
    vec[i] = vec[high];
    vec[high] = tmp;
    // until here.

    // If there are too many busy threads run pretty much a serial sort until threads are available
    if (busythreads >= threads)
    {
        NestedOMPSort2_0(vec, low, i - 1, busythreads, threads);
        NestedOMPSort2_0(vec, i + 1, high, busythreads, threads);
    }
    else
    {
        // Create new threaded recursion with threads if there are any available, slightly different from tasks,
        // instead this is just running parallel code that gets called everytime a thread becomes available.
        
        busythreads += 2;

#pragma omp parallel num_threads(threads) shared(vec, threads, busythreads, i, low, high)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    NestedOMPSort2_0(vec, low, i - 1, busythreads, threads);

                    // this occurs because it will happen after the thread has come about out of the recursive step
                    busythreads--;
                }
#pragma omp section
                {
                    NestedOMPSort2_0(vec, i + 1, high, busythreads, threads);

                    // this occurs because it will happen after the thread has come about out of the recursive step
                    busythreads--;
                }
            }
        }
    }
}

/*
    Starter for Task Queue Quicksort v2.0
    
    int iter = number of times to run the algorithm to get an average run time
    int size = size of the array
    int threads = number of threads to use
    
    return double = elapsed average time
*/
double StartTaskQueueQuickSort2_0(int iter, int size, int threads)
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

// Start the task on a task, and make sure 1 thread is used at first
#pragma omp parallel num_threads(threads) shared(vec)
#pragma omp single
        TaskQueueQuickSort2_0(vec, 0, size - 1);

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

/*
    Task Queue Quicksort v2.0
    
    T *vec = Pointer to an array of template T type
    int low = the low index
    int high = the high index
    std::stack<int> &stack = a shared stack that stores indices of the array
    int &busythreads = a shared counter that determines if the stack can be popped off of
    int &threads = current active threads
*/
template <typename T>
void TaskQueueQuickSort2_0(T *vec, int low, int high)
{
    // Normal serial quicksort
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
        tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
    tmp = vec[i];
    vec[i] = vec[high];
    vec[high] = tmp;
    // Ends here

// Use the openmp task framework to create tasks on a queue
#pragma omp task shared(vec)
    {
        TaskQueueQuickSort2_0(vec, low, i - 1);
    }
#pragma omp task shared(vec)
    {
        TaskQueueQuickSort2_0(vec, i + 1, high);
    }
}

/*
    Starter for PThreads Quicksort v2.0
    
    int iter = number of times to run the algorithm to get an average run time
    int size = size of the array
    int threads = number of threads to use
    
    return double = elapsed average time
*/
double StartPThreadsQuickSort2_0(int iter, int size, int threads)
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

        PThreadsQuickSort2_0(vec, 0, size - 1, activeThreads, threads);

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

/*
    This starts the PThread Quicksort for threads.
    void *param = a void type that is typed casted into a defined struct
*/
void *PThreadsRunner(void *param)
{
    struct pThreadObj<int> p = *(static_cast<pThreadObj<int> *>(param));
    p.activeThreads++;

    PThreadsQuickSort2_0(p.vec, p.low, p.high, p.activeThreads, p.maxThreads);
    return NULL;
}

/*
    PThreads Quicksort v2.0
    
    T *vec = Pointer to an array of template T type
    int low = the low index
    int high = the high index
    int &activeThreads = number of threads that are currently active
    int const maxThreads = total number of threads that can be active at a time
*/
template <typename T>
void PThreadsQuickSort2_0(T *vec, int low, int high, int &activeThreads, int const maxThreads)
{
    // Normal serial quicksort 
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
        tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
    tmp = vec[i];
    vec[i] = vec[high];
    vec[high] = tmp;
    // end serial quicksort

    // If there are threads available assignment
    if (activeThreads < maxThreads)
    {
        // make a thread and struct parameter
        pthread_t thread;
        struct pThreadObj<int> p =
        {
            vec, low, i - 1, activeThreads, maxThreads
        };

        // create a new thread and process it.
        pthread_create(&thread, NULL, PThreadsRunner, &p);
        PThreadsQuickSort2_0(vec, i + 1, high, activeThreads, maxThreads);

        // once the thread comes back out from its recursion join it into the other threads
        pthread_join(thread, NULL);
        activeThreads--;
    }
    else
    {
        // all threads are busy, do a serial sort instead
        PThreadsQuickSort2_0(vec, low, i - 1, activeThreads, maxThreads);
        PThreadsQuickSort2_0(vec, i + 1, high, activeThreads, maxThreads);
    }
}

/*
    Fills an array given the array and size
    
    T* vec = Reference to an array
    int s = size of the array
*/
template <typename T>
void FillArray(T *vec, int s)
{
    // set a seed so the same array is always used.
    srand(777);
    for (int i = 0; i < s; i++)
        vec[i] = rand();
}

/*
    Checks the vector to make sure it is actually sorted
    
    std::vector<T> &vec = reference to the vector that should be sorted
*/
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

/*
    Checks the array to make sure it is actually sorted
    
    T *vec = reference to the array that should be sorted
*/
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

/*
    Output the relevant data from the sorts
    
    double time = time taken to sort the array
    int s = size of the array
    int threads = threads used
    char const name[100] = The name of the function
*/
void Output(double time, int s, int threads, char const name[100])
{

    std::cout << name << std::endl;
    std::cout << "Threads = " << threads << std::endl;
    std::cout << "Array Size = " << s << std::endl;
    std::cout << "Time Elapsed = " << time << std::endl;
}