import numpy as np

half_fast_counter = 0
counter = 0

def half_fast_fourier_transform(f):
    global half_fast_counter
    N = len(f)
    p1 = int(np.sqrt(N))
    p2 = N // p1

    assert p1 * p2 == N, "N должно быть произведением p1 и p2"

    A_1 = np.zeros((p1, p2), dtype=complex)
    for k2 in range(p2):
        for k1 in range(p1):
            summation = 0
            for j1 in range(p1):
                j = j1 + p1 * k2
                half_fast_counter += 1
                exponent = -2j * np.pi * k1 * j1 / p1
                summation += f[j] * np.exp(exponent)
            A_1[k1, k2] = summation

    A_2 = np.zeros((p1, p2), dtype=complex)
    for k1 in range(p1):
        for k2 in range(p2):
            summation = 0
            for j2 in range(p2):
                j = k1 + p1 * j2
                half_fast_counter += 1
                exponent = -2j * np.pi * k2 * j2 / p2
                summation += A_1[k1, j2] * np.exp(exponent)
            A_2[k1, k2] = summation

    A = np.zeros(N, dtype=complex)
    for k1 in range(p1):
        for k2 in range(p2):
            k = k1 + p1 * k2
            A[k] = A_2[k1, k2]

    return A

def inverse_half_fast_fourier_transform(F):
    global half_fast_counter
    N = len(F)
    p1 = int(np.sqrt(N))
    p2 = N // p1

    assert p1 * p2 == N, "N должно быть произведением p1 и p2"

    F_reshaped = np.zeros((p1, p2), dtype=complex)
    for k1 in range(p1):
        for k2 in range(p2):
            k = k1 + p1 * k2
            F_reshaped[k1, k2] = F[k]

    A_inv_1 = np.zeros((p1, p2), dtype=complex)
    for j2 in range(p2):
        for j1 in range(p1):
            summation = 0
            for k2 in range(p2):
                k = j1 + p1 * k2
                exponent = 2j * np.pi * k2 * j2 / p2
                summation += F_reshaped[j1, k2] * np.exp(exponent)
            A_inv_1[j1, j2] = summation

    A_inv_2 = np.zeros((p1, p2), dtype=complex)
    for j1 in range(p1):
        for j2 in range(p2):
            summation = 0
            for k1 in range(p1):
                k = k1 + p1 * j2
                exponent = 2j * np.pi * k1 * j1 / p1
                summation += A_inv_1[k1, j2] * np.exp(exponent)
            A_inv_2[j1, j2] = summation

    A_inv = np.zeros(N, dtype=complex)
    for j1 in range(p1):
        for j2 in range(p2):
            j = j1 + p1 * j2
            A_inv[j] = A_inv_2[j1, j2] / N

    return A_inv

def discrete_fourier_transform(f):
    global counter
    N = len(f)
    F = np.zeros(N, dtype=complex)
    for k in range(N):
        summation = 0
        for n in range(N):
            counter += 1
            exponent = -2j * np.pi * k * n / N
            summation += f[n] * np.exp(exponent)
        F[k] = summation
    return F

def inverse_discrete_fourier_transform(F):
    N = len(F)
    f = np.zeros(N, dtype=complex)
    for n in range(N):
        summation = 0
        for k in range(N):
            exponent = 2j * np.pi * k * n / N
            summation += F[k] * np.exp(exponent)
        f[n] = summation / N
    return f

def func(n):
    array = []
    for i in range(n):
        array.append(i)
    return array

f = np.array(func(1600), dtype=complex)

result_half_fft = half_fast_fourier_transform(f)
inverse_result_half_fft = inverse_half_fast_fourier_transform(result_half_fft)

result_dft = discrete_fourier_transform(f)
inverse_result_dft = inverse_discrete_fourier_transform(result_dft)

print("Результат полубыстрого преобразования Фурье (ПШФ):")
# print(result_half_fft)
print(half_fast_counter)

print("\nРезультат обратного полубыстрого преобразования Фурье:")
# print(inverse_result_half_fft)

print("\nРезультат обычного преобразования Фурье (DFT):")
# print(result_dft)
print(counter)

print("\nРезультат обратного преобразования Фурье (IDFT):")
# print(inverse_result_dft)
