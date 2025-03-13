#include "include.h"

#define VECTOR_SIZE 100000

int main(int argc, char* argv[])
{
    std::mt19937 gen(std::random_device{}());
    std::vector<int> vec(VECTOR_SIZE);

    int min_val = 1;
    int max_val = 100;
    std::uniform_int_distribution<int> dist(min_val, max_val);

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

    #pragma region QuickSort

    copy = vec; 
    start = std::chrono::high_resolution_clock::now();
    std::sort(vec.begin(), vec.end(), [](int a, int b){return a < b;});
    end = std::chrono::high_resolution_clock::now();

    duration = end - start;

    std::cout << '[' << vec.size() << ']' << " Quick:  " << duration.count() << " seconds\n"; 

    #pragma endregion

    return 0;
}