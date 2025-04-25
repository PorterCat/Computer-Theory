#include <vector>
#include <algorithm>
#include <iostream>

std::vector<double> convolution(const std::vector<double>& signal, const std::vector<double>& filter) 
{
    //std::vector<int> filter(filter.rbegin(), filter.rend());
    
    int output_len = signal.size() - filter.size() + 1;
    std::vector<double> result(output_len, 0.0);

    for (int i = 0; i < output_len; ++i) 
    {
        double sum = 0.0;
        for (size_t j = 0; j < filter.size(); ++j) 
        {
            sum += signal[i + j] * filter[j];
        }
        result[i] = sum;
    }

    return result;
}

int main(int argc, char* argv[]) 
{
    std::vector<double> signal = {1, 2, 3, 4, 5};
    std::vector<double> filter = {-1, 0, 1};

    std::vector<double> result = convolution(signal, filter);

    std::cout << "Result: ";
    for (double val : result) 
        std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}