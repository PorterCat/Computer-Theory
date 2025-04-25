#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <cassert>

int dft_counter = 0;

struct Complex 
{
    double re;
    double im;
    
    Complex() : re(0), im(0) {}
    Complex(double real, double imag = 0) : re(real), im(imag) {}
};

Complex operator* (const Complex& a, const Complex& b) {
    return Complex(
        a.re * b.re - a.im * b.im, 
        a.re * b.im + a.im * b.re
    );
}

Complex operator+ (const Complex& a, const Complex& b) {
    return Complex(a.re + b.re, a.im + b.im);
}

Complex operator/ (const Complex& a, double divisor) {
    return {a.re / divisor, a.im / divisor};
}

Complex complex_exp(double angle) {
    return Complex(cos(angle), sin(angle));
}

std::vector<Complex> dft(const std::vector<double>& f) 
{
    int N = f.size();
    std::vector<Complex> result(N);
    
    for (int k = 0; k < N; ++k) 
    {
        Complex sum;
        for (int n = 0; n < N; ++n) 
        {
            dft_counter++;
            double angle = -2 * M_PI * k * n / N;
            Complex term = Complex(f[n]) * complex_exp(angle);
            sum = sum + term;
        }
        result[k] = sum;
    }
    
    return result;
}

std::vector<Complex> idft(const std::vector<Complex>& f) 
{
    int N = f.size();
    std::vector<Complex> result(N);
    
    for (int n = 0; n < N; ++n) 
    {
        Complex sum;
        for (int k = 0; k < N; ++k) 
        {
            double angle = 2 * M_PI * k * n / N;
            Complex term = f[k] * complex_exp(angle);
            sum = sum + term;
        }
        result[n] = sum / N;
    }
    
    return result;
}