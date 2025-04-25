#include "DicretFourier.hpp"
#include <chrono>

std::vector<Complex> generate_signal(int N) 
{
    std::vector<Complex> signal;
    for (int i = 0; i < N; ++i) 
    {
        signal.push_back(static_cast<double>(i));
    }
    return signal;
}

int main(int argc, char* argv[])
{
    int N = 2500;
    auto signal = generate_signal(N);

    auto start = std::chrono::high_resolution_clock::now();
    auto result_dft = dft(signal);
    auto inverse_dft = idft(result_dft);
    auto end = std::chrono::high_resolution_clock::now();
    auto dft_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Results for N = " << N << ":\n";
    std::cout << "\nOperations: " << dft_counter << "\n";
    std::cout << "Time: " << dft_time << " mc\n";

    return 0;
}