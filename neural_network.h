#include <vector>
#include <string>
#include "tensor.h"

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

struct Operation {
    Tensor weights;
    Tensor* input;
    Tensor output;
    Tensor gradOperation; //gradient of the output with respect to the input
    Tensor gradWeight; //gradient of the output with respect to the weight
    std::string operationType;
    Operation(std::string& operationType);
};

class NeuralNetwork {
   public:
    std::vector<Operation> sequence;

    NeuralNetwork(std::vector<Operation> sequence);

    void backpropagate();
};

#endif