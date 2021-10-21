#include "operators.h"
#include <cassert>
#include <cmath>

Operation::Operation(Tensor* input) {
    this->input = input;
}

MatrixMult::MatrixMult(Tensor* input, Tensor weights)
    : Operation(input) {
    this->weights = weights;
    this->output = Tensor(0., input->shape);
    this->gradWeights = Tensor(0., input->shape);
}

void MatrixMult::execute() {
    //assert(out.shape == resultShape(input->shape, output.shape)); TODO: fix
    int row1 = weights.shape[0];
    int row2 = input->shape[0];
    int col1 = weights.shape[1];
    int col2 = input->shape[1];
    for (int r = 0; r < row1; r++) {
        for (int ansC = 0; ansC < col2; ansC++) {
            float dot = 0;
            for (int c = 0; c < col1; c++) {
                dot += weights.array[r * col1 + c] * input->array[c * col2 + ansC];
            }
            output.array[r * col2 + ansC] = dot;
        }
    }
}

void MatrixMult::gradOp() {
}

void MatrixMult::gradW() {
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

MSE::MSE(Tensor* input, Tensor actual)
    : Operation(input) {
    this->actual = actual;
    this->output = Tensor(0., {1});
    this->gradOperation = Tensor(0., {actual.length, 1});
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