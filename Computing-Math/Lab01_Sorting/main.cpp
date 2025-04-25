#include "include.h"

int main(int argc, char* argv[])
{
    std::mt19937 gen(std::random_device{}());
    std::vector<int> vec;

    uint32_t vector_size = 10;
    if(argc == 2) 
    {
        vector_size = std::stoi(argv[1]);
        vec.resize(vector_size);
    }
    else 
    {
        vec.resize(vector_size);
    }

    std::uniform_int_distribution<int> dist((vector_size == 10) ? 0 : INT32_MIN, 
                                            (vector_size == 10) ? 10 : INT32_MAX);

    for(auto& num : vec)
        num = dist(gen);

    auto clockNow = &std::chrono::high_resolution_clock::now;

    std::vector<int> copy = vec; 
    auto start = clockNow();
    mysorts::bubbleSort(copy, [](int a, int b){return a < b;});
    auto end = clockNow();

    std::chrono::duration<double> duration = end - start;

    std::cout << '[' << vec.size() << ']' << " Bubble: " << duration.count() << " seconds\n";   
    
    #pragma region SelectSort
    
    copy = vec;
    start = clockNow();
    mysorts::selectSort(copy, [](int a, int b){return a < b;});
    end = clockNow();
    duration = end - start;

    std::cout << '[' << vec.size() << ']' << " Select: " << duration.count() << " seconds\n"; 

    #pragma endregion

    #pragma region MergeSort

    copy = vec; 
    start = clockNow();
    mysorts::mergeSort(copy, [](int a, int b){return a < b;}, (vector_size == 10));
    end = clockNow();

    duration = end - start;

    std::cout << '[' << vec.size() << ']' << " Merge:  " << duration.count() << " seconds\n"; 

    #pragma endregion

    return 0;
}

// TODO: попробовать запускать все сортировки в разных потоках - для интереса