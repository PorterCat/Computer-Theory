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
}