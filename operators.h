#include "neural_network.h"

class MatrixMult : public Operation {
    Tensor weights;

   public:
    MatrixMult(Tensor* input, Tensor weights);
    void execute();
    void gradOp();
    void gradW();
};

class MSE : public Operation {
    Tensor actual;

   public:
    MSE(Tensor* input, Tensor actual);
    void execute();
    void gradOp();
    Tensor getGradOp();
};