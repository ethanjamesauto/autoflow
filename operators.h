#include "neural_network.h"

#ifndef OPERATORS_H
#define OPERATORS_H

class MatrixMult : public WeightedOperation {
   public:
    Tensor learningRate;
    MatrixMult(Tensor* input, int outputLength);
    void execute();
    Tensor getGradOp();
    Tensor getGradWeights();
};

class MatrixAdd : public WeightedOperation {
   public:
    Tensor learningRate;
    MatrixAdd(Tensor* input);
    void execute();
    Tensor getGradOp();
    Tensor getGradWeights();
};

class Relu : public Operation {
    Tensor gradOperation;

   public:
    Relu(Tensor* input);
    void execute();
    void gradOp();
    Tensor getGradOp();
};

class Sigmoid : public Operation {
    Tensor gradOperation;

   public:
    Sigmoid(Tensor* input);
    void execute();
    void gradOp();
    Tensor getGradOp();
};

class MSE : public Operation {
    Tensor gradOperation;

   public:
    Tensor* actual;
    MSE(Tensor* input, Tensor* actual);
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