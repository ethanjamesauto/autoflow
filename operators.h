#include "neural_network.h"

struct MatrixMult : Operation {
    Tensor weights;
    void gradOp();
    void gradW();
};