#include "include.h"

#define VECTOR_SIZE 1000

int main(int argc, char* argv[])
{
    std::mt19937 gen(std::random_device{}());
    std::vector<int> vec(VECTOR_SIZE);

    std::uniform_int_distribution<int> dist(INT32_MIN, INT32_MAX);

    for(auto& num : vec)
        num = dist(gen);

    std::vector<int> copy = vec; 
    auto start = std::chrono::high_resolution_clock::now();
    mysorts::bubbleSort(copy, [](int a, int b){return a < b;});
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << '[' << vec.size() << ']' << " Bubble: " << duration.count() << " seconds\n";   
    
    #pragma region SelectSort
    
    copy = vec;
    start = std::chrono::high_resolution_clock::now();
    mysorts::selectSort(copy, [](int a, int b){return a < b;});
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    std::cout << '[' << vec.size() << ']' << " Select: " << duration.count() << " seconds\n"; 

    #pragma endregion

    #pragma region MergeSort

    copy = vec; 
    start = std::chrono::high_resolution_clock::now();
    mysorts::mergeSort(copy, [](int a, int b){return a < b;});
    end = std::chrono::high_resolution_clock::now();

    duration = end - start;

    std::cout << '[' << vec.size() << ']' << " Merge:  " << duration.count() << " seconds\n"; 

    #pragma endregion

    return 0;
}