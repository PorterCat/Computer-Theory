#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <random>

template <typename Container>
void printContainer(const Container& container, const std::string& delimiter = " ", 
    const std::string& prefix = "[", const std::string& suffix = "]") 
{
    std::cout << prefix;
    for (auto it = container.begin(); it != container.end(); ++it) 
    {
        std::cout << *it;
        if (std::next(it) != container.end())
            std::cout << delimiter;
    }
    std::cout << suffix << std::endl;
}

template<typename T, typename Compare>
void merge(std::vector<T>& arr, Compare comp, int left, int mid, int right) 
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<T> L(n1), R(n2);

    for (size_t i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (size_t j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) 
    {
        if (L[i] <= R[j]) 
            arr[k++] = L[i++];
        else 
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];
}

namespace mysorts
{
    template<typename T, typename Compare>
    void bubbleSort(std::vector<T>& vec, Compare comp)
    {
        for(size_t i = 0; i < vec.size(); ++i)
            for(size_t j = i + 1; j < vec.size(); ++j)
                if(comp(vec[j], vec[i]))
                    std::swap(vec[i], vec[j]);
    }

    template<typename T, typename Compare>
    void selectSort(std::vector<T>& vec, Compare comp)
    {
        for(size_t i = 0; i < vec.size(); ++i)
        {
            size_t changeIndex = i;
            for(size_t j = i + 1; j < vec.size(); ++j)
            {
                if(comp(vec[j], vec[changeIndex]))
                    changeIndex = j;
            }
            std::swap(vec[i], vec[changeIndex]);
        }
    }

    template<typename T, typename Compare>
    void mergeSort(std::vector<T>& vec, Compare comp)
    {
        int n = vec.size();

        for (int currSize = 1; currSize <= n - 1; currSize *= 2) 
        {
            for (int leftStart = 0; leftStart < n - 1; leftStart += 2 * currSize) 
            {
                int mid = std::min(leftStart + currSize - 1, n - 1);
                int rightEnd = std::min(leftStart + 2 * currSize - 1, n - 1);

                merge(vec, comp, leftStart, mid, rightEnd);
            }
        }
    }
}