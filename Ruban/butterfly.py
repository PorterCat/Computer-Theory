import cmath
import math

def fft(signal):
    n = len(signal)
    if n <= 1:
        return signal, 0

    even, even_mults = fft(signal[0::2])
    odd, odd_mults = fft(signal[1::2])

    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    mult_count = n // 2 

    combined = [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]
    return combined, even_mults + odd_mults + mult_count

if __name__ == "__main__":
    signal = [i for i in range(1600)]
    next_power_of_2 = 2 ** math.ceil(math.log2(len(signal)))
    signal += [0] * (next_power_of_2 - len(signal))

    result, total_mults = fft(signal)
    # print("FFT результат:", result)
    print("Общее количество умножений:", total_mults)
