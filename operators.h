#include "neural_network.h"

#ifndef OPERATORS_H
#define OPERATORS_H

class MatrixMult : public Operation {
    Tensor gradWeights;

   public:
    Tensor weights;
    MatrixMult(Tensor* input, Tensor weights);
    void execute();
    Tensor getGradOp();
    Tensor getGradWeights();
};

class MatrixAdd : public Operation {
   public:
    Tensor weights;
    MatrixAdd(Tensor* input, Tensor weights);
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

class MSE : public Operation {
    Tensor* actual;
    Tensor gradOperation;

   public:
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