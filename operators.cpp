#include "operators.h"

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

void MSE::execute() {
}

void MSE::gradOp() {
}