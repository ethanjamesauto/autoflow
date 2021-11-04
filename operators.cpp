#include "operators.h"
#include <cassert>
#include <cmath>

MatrixMult::MatrixMult(Tensor* input, int outputLength)
    : WeightedOperation(input) {
    this->weights = Tensor::random({outputLength, input->length});
    this->learningRate = Tensor(0, weights.shape);
    this->output = Tensor({outputLength, 1});
}

void MatrixMult::execute() {
    Tensor::matmult(weights, *input, output);
}

Tensor MatrixMult::getGradOp() {
    return weights;
}
Tensor MatrixMult::getGradWeights() {
    //Tensor ret = *input;
    //ret.shape = {1, ret.length};
    //return ret;
    return *input;
}

MatrixAdd ::MatrixAdd(Tensor* input)
    : WeightedOperation(input) {
    this->weights = Tensor::random(input->shape);
    this->learningRate = Tensor(0, weights.shape);
    this->output = Tensor(input->shape);
}

void MatrixAdd::execute() {
    Tensor::add(*input, weights, output);
}

Tensor MatrixAdd::getGradOp() {
    return Tensor(1, input->shape);
}

Tensor MatrixAdd::getGradWeights() {
    return Tensor(1, input->shape);
}

MSE::MSE(Tensor* input, Tensor* actual)
    : Operation(input) {
    this->actual = actual;
    this->output = Tensor({1});
    this->gradOperation = Tensor({actual->length, 1});
}

void MSE::execute() {
    output.array[0] = Tensor::mse(*input, *actual);
}

void MSE::gradOp() {
    Tensor& exp = *input;
    assert(exp.shape == actual->shape);
    float scalar = 2. / exp.length;
    for (int i = 0; i < exp.length; i++) {
        gradOperation.array[i] = scalar * (exp.array[i] - actual->array[i]);
    }
}

Tensor MSE::getGradOp() {
    return gradOperation;
}
