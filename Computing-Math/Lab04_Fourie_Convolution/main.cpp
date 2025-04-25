#include <iostream>
#include "../Lab02_DPFn2/DicretFourier.hpp"

std::vector<double> convolution_dft(const std::vector<double>& signal, const std::vector<double>& filter) 
{
    int n = signal.size() + filter.size() - 1;
    
    std::vector<double> padded_signal = signal;
    padded_signal.resize(n);
    std::vector<double> padded_kernel = filter;
    padded_kernel.resize(n);
    
    std::vector<Complex> signal_dft = dft(padded_signal);
    std::vector<Complex> kernel_dft = dft(padded_kernel);
    
    std::vector<Complex> result_dft(n);
    for (int i = 0; i < n; ++i) 
    {
        result_dft[i] = signal_dft[i] * kernel_dft[i];
    }
    
    std::vector<Complex> result_complex = idft(result_dft);
    std::vector<double> result;
    for (const auto& val : result_complex) 
    {
        result.push_back(val.re);
    }
    
    return result;
}

int main()
{
    std::vector<double> signal = {1, 2, 3, 4, 5};
    std::vector<double> filter = {1, 0, -1};
    
    std::vector<double> result = convolution_dft(signal, filter);
    
    std::cout << "Result ";
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << '\n';

    return 0;
}

