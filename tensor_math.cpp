#include "tensor.h"

#include <cassert>
#include <cmath>

constexpr float max(float a, float b) {
    return a > b ? a : b;
}

std::vector<int> resultShape(std::vector<int> a, std::vector<int> b) {
    assert(a[1] == b[0]);
    std::vector<int> ret{a[0], b[1]};
    return ret;
}

void Tensor::reluMutable() {
    for (int i = 0; i < length; i++) {
        array[i] = max(array[i], 0);
    }
}

Tensor Tensor::relu() {
    Tensor t(*this);
    t.reluMutable();
    return t;
}

void Tensor::addMutable(Tensor& other) {
    assert(shape == other.shape);
    for (int i = 0; i < length; i++) {
        array[i] += other.array[i];
    }
}

void Tensor::softmaxMutable() {
    double sum;
    for (int i = 0; i < length; i++) {
        array[i] = exp(array[i]);
        sum += array[i];
    }
    for (int i = 0; i < length; i++) {
        array[i] /= sum;
    }
}

Tensor Tensor::softmax() {
    Tensor t(*this);
    t.softmaxMutable();
    return t;
}

void Tensor::matmult(Tensor& one, Tensor& two, Tensor& result_container) {
    assert(result_container.shape == resultShape(one.shape, two.shape));
    int row1 = one.shape[0];
    int row2 = two.shape[0];
    int col1 = one.shape[1];
    int col2 = two.shape[1];
    for (int r = 0; r < row1; r++) {
        for (int ansC = 0; ansC < col2; ansC++) {
            float dot = 0;
            for (int c = 0; c < col1; c++) {
                dot += one.array[r * col1 + c] * two.array[c * col2 + ansC];
            }
            result_container.array[r * col2 + ansC] = dot;
        }
    }
}

Tensor Tensor::matmult(Tensor& one, Tensor& two) {
    Tensor result = Tensor(0., resultShape(one.shape, two.shape));
    matmult(one, two, result);
    return result;
}