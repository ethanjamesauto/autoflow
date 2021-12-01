#include "operators.h"
#include <cmath>

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

float sigmoid(float x) {
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

Softmax::Softmax(Tensor* input)
    : Operation(input) {
    this->output = Tensor(input->shape);
    this->gradOperation = Tensor({input->length, input->length});
}

void Softmax::execute() {
    Tensor::softmax(*input, output);
}

void Softmax::gradOp() {
    Tensor& z = *input; //TODO: see if there's a difference between this and directly referencing input
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
