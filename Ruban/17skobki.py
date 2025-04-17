import sys


def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = sys.maxsize
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]

                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m, s


def get_optimal_parens(s, i, j):
    if i == j:
        return f"A{i+1}"

    else:
        left = get_optimal_parens(s, i, s[i][j])
        right = get_optimal_parens(s, s[i][j] + 1, j)
        print(f"({left} x {right})")
        return f"({left} x {right})"


p = [10, 20, 50, 1, 100, 100, 100, 100, 100]
m, s = matrix_chain_order(p)
print(f"(M: {m} S: {s})")

optimal_order = get_optimal_parens(s, 0, len(p) - 2)
min_operations = m[0][len(p) - 2]

print(f"Минимальное количество скалярных умножений: {min_operations}")
print(f"Оптимальный порядок умножения матриц: {optimal_order}")
