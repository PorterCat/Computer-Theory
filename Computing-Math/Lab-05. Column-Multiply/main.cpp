#include <iostream>
#include <vector>
#include <string>

std::string multiply_strings(const std::string& num1, const std::string& num2) 
{
    if (num1 == "0" || num2 == "0") {
        return "0";
    }

    int n1 = num1.size();
    int n2 = num2.size();
    std::vector<int> result(n1 + n2, 0);

    for (int i = n1 - 1; i >= 0; i--) 
    {
        for (int j = n2 - 1; j >= 0; j--)
        {
            int digit1 = num1[i] - '0';
            int digit2 = num2[j] - '0';
            int mul = digit1 * digit2;
            int p2 = i + j + 1;
            int p1 = i + j;

            int total = mul + result[p2];
            result[p2] = total % 10;
            result[p1] += total / 10;

            while (p1 >= 0 && result[p1] > 9) 
            {
                int carry = result[p1] / 10;
                result[p1] %= 10;
                p1--;
                if (p1 >= 0)
                    result[p1] += carry;
            }
        }
    }

    std::string result_str;
    int idx = 0;
    while (idx < result.size() && result[idx] == 0)
        idx++;

    if (idx == result.size()) 
        return "0";

    for (; idx < result.size(); idx++) 
        result_str += (result[idx] + '0');

    return result_str;
}

int main(int argc, char* argv[])
{
    if (argc != 3) 
    {
        std::cerr << "Usage: " << argv[0] << " <num1> <num2>" << std::endl;
        return 1;
    }

    std::string num1 = std::string(argv[1]);
    std::string num2 = std::string(argv[2]);
    std::cout << multiply_strings(num1, num2) << std::endl;
    return 0;
}