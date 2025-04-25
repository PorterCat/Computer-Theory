#include <climits>
#include <iostream>
#include <vector>

using matrix = std::vector<std::vector<int>>;

void printOptimalParens(const matrix& s, int i, int j) 
{
    if (i == j) 
    {
        std::cout << "A" << i;
    } 
    else 
    {
        std::cout << "(";
        printOptimalParens(s, i, s[i][j]);
        printOptimalParens(s, s[i][j] + 1, j);
        std::cout << ")";
    }
}

int main(int argc, char* argv[]) 
{
    std::vector<int> p = {10, 20, 50, 1, 100};

    int n = p.size();
    matrix m(n, std::vector<int>(n, 0));
    matrix s(n, std::vector<int>(n, 0));

    for (int len = 2; len < n; ++len)
    {
        for (int i = 1; i < n - len + 1; ++i)
        {
            int j = i + len - 1;
            m[i][j] = INT_MAX;

            for (int k = i; k < j; ++k)
            {
                int q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];

                printf("m[%d][%d] = %d, m[%d][%d] = %d, m[%d][%d] = %d\n", i, j,
                    m[i][j], i, k, m[i][k], k + 1, j, m[k + 1][j]);

                if (q < m[i][j])
                {
                    printf("Swapped: m[%d][%d] = %d, m[%d][%d] = %d, m[%d][%d] = %d\n", i,
                        j, m[i][j], i, k, m[i][k], k + 1, j, m[k + 1][j]);
                        
                    m[i][j] = q;
                    s[i][j] = k;
                }
            }
        }
    }

    std::cout << "Minimum number of multiplications: " << m[1][n - 1] << std::endl;
    std::cout << "Optimal Parenthesization: ";
    printOptimalParens(s, 1, n - 1);
    std::cout << std::endl;

    return 0;
}