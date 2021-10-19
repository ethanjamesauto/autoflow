#include "neural_network.h"

struct MatrixMult : Operation {
    Tensor weights;
    MatrixMult(Tensor* input, Tensor weights);
    void execute();
    void gradOp();
    void gradW();
};

struct MSE : Operation {
    MSE(Tensor* input)
        : Operation(input) {}
    void execute();
    void gradOp();
};