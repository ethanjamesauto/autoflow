#include "neural_network.h"

struct MatrixMult : Operation {
    MatrixMult(Tensor* input, Tensor weights);
    void execute();
    void gradOp();
    void gradW();
};

struct MSE : Operation {
    Tensor actual;
    MSE(Tensor* input, Tensor actual);
    void execute();
    void gradOp();
};