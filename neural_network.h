#include <vector>
#include <string>
#include "tensor.h"

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

struct Operation {
    Tensor* input; //Note: be careful with this pointer! Stay away from new.
    Tensor output;
    Tensor gradOperation; //gradient of the output with respect to the input
    Tensor gradWeight; //gradient of the output with respect to the weight
    virtual void gradOp() {}
    virtual void gradW() {}
};

class NeuralNetwork {
   public:
    std::vector<Operation> sequence;

    NeuralNetwork(std::vector<Operation> sequence);

    void backpropagate();
};

#endif