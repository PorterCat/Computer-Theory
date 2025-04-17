#include <climits>
#include <iostream>

using namespace std;

void printOptimalParens(int **s, int i, int j) {
  if (i == j) {
    cout << "A" << i;
  } else {
    cout << "(";
    printOptimalParens(s, i, s[i][j]);
    printOptimalParens(s, s[i][j] + 1, j);
    cout << ")";
  }
}

void matrixChainOrder(int *p, int n) {
  int **m = new int *[n];

  int **s = new int *[n];

  for (int i = 0; i < n; ++i) {
    m[i] = new int[n];
    s[i] = new int[n];
  }

  for (int i = 1; i < n; ++i) {
    m[i][i] = 0;
  }

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

          printf("Swapped: m[%d][%d] = %d, m[%d][%d] = %d, m[%d][%d] = %d\n", i,
                 j, m[i][j], i, k, m[i][k], k + 1, j, m[k + 1][j]);
        {
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }

  cout << "Minimum number of multiplications: " << m[1][n - 1] << endl;

  cout << "Optimal Parenthesization: ";
  printOptimalParens(s, 1, n - 1);
  cout << endl;

  for (int i = 0; i < n; ++i) {
    delete[] m[i];
    delete[] s[i];
  }
  delete[] m;
  delete[] s;
}

int main() {
  int p[] = {10, 100, 5, 50};
  int n = sizeof(p) / sizeof(p[0]);

  matrixChainOrder(p, n);

  return 0;
}
