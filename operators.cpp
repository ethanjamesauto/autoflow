#include "operators.h"
#include <cassert>
#include <cmath>

Operation::Operation(Tensor* input) {
    this->input = input;
}

Operation::Operation() {}

MatrixMult::MatrixMult(Tensor* input, Tensor weights)
    : Operation(input) {
    this->weights = weights;
    this->output = Tensor(input->shape);
    this->gradWeights = Tensor(input->shape);
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

MatrixAdd ::MatrixAdd(Tensor* input, Tensor weights)
    : Operation(input) {
    this->weights = weights;
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

Relu::Relu(Tensor* input)
    : Operation(input) {
    this->output = Tensor(input->shape);
    this->gradOperation = Tensor(input->shape);
}

void Relu::execute() {
    Tensor::relu(*input, output);
}

void Relu::gradOp() {
    for (int i = 0; i < input->length; i++) {
        gradOperation.array[i] = input->array[i] > 0 ? 1 : 0;
    }
}

Tensor Relu::getGradOp() {
    return gradOperation;
}

float sigmoid (float x) {
    return 1 / (1 + exp(-x));
}

Sigmoid::Sigmoid(Tensor* input)
    : Operation(input) {
    this->output = Tensor(input->shape);
    this->gradOperation = Tensor(input->shape);
}

void Sigmoid::execute() {
    for (int i = 0; i < input->length; i++) {
        output.array[i] = sigmoid(input->array[i]);
    }
}

void Sigmoid::gradOp() {
    for (int i = 0; i < input->length; i++) {
        float x = output.array[i];
        gradOperation.array[i] = x * (1 - x);
    }
}

Tensor Sigmoid::getGradOp() {
    return gradOperation;
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

Softmax::Softmax(Tensor* input)
    : Operation(input) {
    this->output = Tensor(input->shape);
    this->gradOperation = Tensor({input->length, input->length});
}

void Softmax::execute() {
    Tensor::softmax(*input, output);
}

//TODO: don't make an nxn tensor; for optizimation
void Softmax::gradOp() {
    Tensor& z = *input;
    float x = 0;
    for (int i = 0; i < z.length; i++) {
        x += exp(z.array[i]);
    }
    for (int k = 0; k < z.length; k++) {
        for (int l = 0; l < z.length; l++) {
            float& val = gradOperation.array[k * z.length + l];
            val = 0;
            if (k == l) {
                val += x * exp(z.array[k]);
            }
            val -= exp(z.array[k]) * exp(z.array[l]);
            val /= x * x;
        }
    }
}

Tensor Softmax::getGradOp() {
    return gradOperation;
}