#include "neural_network.h"

#ifndef OPERATORS_H
#define OPERATORS_H

class MatrixMult : public Operation {
    Tensor weights;
    Tensor gradWeights;

   public:
    MatrixMult(Tensor* input, Tensor weights);
    void execute();
    void gradOp();
    void gradW();
    Tensor getGradOp();
    Tensor getGradWeights();
};

class MSE : public Operation {
    Tensor actual;
    Tensor gradOperation;

   public:
    MSE(Tensor* input, Tensor actual);
    void execute();
    void gradOp();
    Tensor getGradOp();
};

class Softmax : public Operation {
    Tensor gradOperation;

   public:
    Softmax(Tensor* input);
    void execute();
    void gradOp();
    Tensor getGradOp();
};

#endif