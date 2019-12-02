#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#define INSERT_THRESH 4

template <typename T>
void SerialInsertionSort(std::vector<T> &vec, int l, int r);
template <typename T>
void SerialQuickSort(std::vector<T> &vec, int l, int r);

int main()
{
    std::vector<int> vec;
    int64_t elapsed = 0;
    for (int i = 0; i < 10; i++)
    {
        vec.clear();
        int n = 123456;
        for (int i = 0; i < n; i++)
        {
            vec.push_back(rand() % n + n);
        }

        auto start = std::chrono::high_resolution_clock::now();
        SerialQuickSort(vec, 0, vec.size() - 1);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        elapsed += duration.count();
    }

    elapsed /= 10;

    int leng = vec.size();
    bool isSorted = true;
    for (int i = 1; i < leng; i++)
    {
        if (vec[i - 1] > vec[i])
            isSorted = false;
    }

    std::cout << "Sorted = " << isSorted << std::endl;
    std::cout << "duration = " << elapsed << std::endl;
}

template <typename T>
void SerialQuickSort(std::vector<T> &vec, int l, int r)
{
    T pivot;
    T tmp;
    int i, j;

    if (r - l < INSERT_THRESH) // hard code for now
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

    return;
}