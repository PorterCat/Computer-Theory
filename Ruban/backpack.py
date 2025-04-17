def knapsack(n, W, a, c):
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    items_used = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(W + 1):
            dp[i][j] = dp[i - 1][j]
            if a[i] <= j:
                new_value = dp[i][j - a[i]] + c[i]
                if new_value > dp[i][j]:
                    dp[i][j] = new_value
                    items_used[i][j] = items_used[i][j - a[i]] + 1

    counts = [0] * (n + 1)
    weight_left = W
    for i in range(n, 0, -1):
        while (
            weight_left >= a[i]
            and dp[i][weight_left] == dp[i][weight_left - a[i]] + c[i]
        ):
            counts[i] += 1
            weight_left -= a[i]

    return dp[n][W], counts


if __name__ == "__main__":
    n = 3
    W = 1000000

    a = [0, 3, 5, 8]
    c = [0, 8, 14, 23]

    max_value, counts = knapsack(len(a) - 1, W, a, c)
    print(f"Max price: {max_value}")
    print("Items used:")
    for i in range(1, len(counts)):
        print(f"Item {i}: {counts[i]} times")
