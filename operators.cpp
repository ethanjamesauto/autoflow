#include "operators.h"
#include <cassert>

Operation::Operation(Tensor* input) {
    this->input = input;
}

MatrixMult::MatrixMult(Tensor* input, Tensor weights)
    : Operation(input) {
    this->weights = weights;
}

void MatrixMult::execute() {
}

void MatrixMult::gradOp() {
}

void MatrixMult::gradW() {
}

MSE::MSE(Tensor* input, Tensor actual)
    : Operation(input) {
    this->actual = actual;
    this->output = Tensor(0., {1});
    this->gradOperation = Tensor(0., actual.shape);
}

void MSE::execute() {
    output.array[0] = Tensor::mse(*input, actual);
}

void MSE::gradOp() {
    Tensor& exp = *input;
    assert(exp.shape == actual.shape);
    float scalar = 2. / exp.length;
    for (int i = 0; i < exp.length; i++) {
        gradOperation.array[i] = scalar * (exp.array[i] - actual.array[i]);
    }
}