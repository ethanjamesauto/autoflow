#include "operators.h"
#include <cassert>
#include <cmath>

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
    this->gradOperation = Tensor(0., {1, actual.length});
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

Tensor MSE::getGradOp() {
    return gradOperation;
}

Softmax::Softmax(Tensor* input)
    : Operation(input) {
    this->output = Tensor(0., input->shape);
    this->gradOperation = Tensor(0., {input->length, input->length});
}

void Softmax::execute() {
    double sum;
    for (int i = 0; i < input->length; i++) {
        output.array[i] = exp(input->array[i]);
        sum += output.array[i];
    }
    for (int i = 0; i < input->length; i++) {
        output.array[i] /= sum;
    }
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