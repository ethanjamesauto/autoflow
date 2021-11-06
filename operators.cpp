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

CategoricalCrossEntropy::CategoricalCrossEntropy(Tensor* input, Tensor* actual)
    : Operation(input) {
    this->actual = actual;
    this->output = Tensor({1});
    this->gradOperation = Tensor({actual->length, 1});
}

void CategoricalCrossEntropy::execute() {
    output.array[0] = 0;
    assert(input->shape == actual->shape);
    for (int i = 0; i < input->length; i++) {
        output.array[0] -= actual->array[i] * log(input->array[i]);
    }
}

/**
 * @brief WARNING: this method combines the gradient of this operation with the softmax function!
 * If you're also using the softmax function, there's no need to use it's gradient. If you're not
 * using the softmax function, you probably shouldn't be using this loss function in the first place.
 * 
 */
void CategoricalCrossEntropy::gradOp() {
    Tensor tmp = actual->scalarMult(-1);
    for (int i = 0; i < input->length; i++) {
        gradOperation.array[i] = input->array[i] - actual->array[i];
    }
}

Tensor CategoricalCrossEntropy::getGradOp() {
    return gradOperation;
}