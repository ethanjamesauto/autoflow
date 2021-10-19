#include "tensor.h"

#include <cassert>
#include <cmath>

constexpr float max(const float a, const float b) {
    return a > b ? a : b;
}

std::vector<int> resultShape(const std::vector<int>& a, const std::vector<int>& b) {
    assert(a[1] == b[0]);
    std::vector<int> ret{a[0], b[1]};
    return ret;
}

void Tensor::reluMutable() {
    for (int i = 0; i < length; i++) {
        array[i] = max(array[i], 0);
    }
}

Tensor Tensor::relu() const {
    Tensor t(*this);
    t.reluMutable();
    return t;
}

void Tensor::addMutable(const Tensor& other) {
    assert(shape == other.shape);
    for (int i = 0; i < length; i++) {
        array[i] += other.array[i];
    }
}

Tensor Tensor::add(const Tensor& other) const {
    Tensor t(*this);
    t.addMutable(other);
    return t;
}

void Tensor::scalarMultMutable(float scalar) {
    for (int i = 0; i < length; i++) {
        array[i] *= scalar;
    }
}

Tensor Tensor::scalarMult(float scalar) {
    Tensor t(*this);
    t.scalarMultMutable(scalar);
    return t;
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

Tensor Tensor::softmax() const {
    Tensor t(*this);
    t.softmaxMutable();
    return t;
}

void Tensor::matmult(const Tensor& one, const Tensor& two, Tensor& out) {
    assert(out.shape == resultShape(one.shape, two.shape));
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
            out.array[r * col2 + ansC] = dot;
        }
    }
}

Tensor Tensor::matmult(const Tensor& one, const Tensor& two) {
    Tensor result = Tensor(0., resultShape(one.shape, two.shape));
    matmult(one, two, result);
    return result;
}

void Tensor::outer_product(const Tensor& one, const Tensor& two, Tensor& out) {
    assert(one.shape.size() == 2 && two.shape.size() == 2);
    assert(one.shape[1] == 1 && two.shape[1] == 1);
    assert(out.shape[0] == one.shape[0] && out.shape[1] == two.shape[0]);
    for (int r = 0; r < out.shape[0]; r++) {
        for (int c = 0; c < out.shape[1]; c++) {
            out.array[r * out.shape[1] + c] = one.array[r] * two.array[c];
        }
    }
}

float Tensor::mse(const Tensor& exp, const Tensor& actual) {
    assert(exp.shape == actual.shape);
    float ans = 0;
    for (int i = 0; i < exp.length; i++) {
        ans += std::pow(actual.array[i] - exp.array[i], 2.);
    }
    return ans / exp.length;
}