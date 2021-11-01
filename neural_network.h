#include <string>
#include <vector>
#include "tensor.h"

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

class Operation {
   public:
    Tensor* input = NULL;  //Note: be careful with this pointer! Stay away from new.
    Tensor output;
    Operation(Tensor* input);
    Operation();
    virtual void execute() {}
    virtual void gradOp() {}
    virtual Tensor getGradOp() { return Tensor(); }
};

class WeightedOperation : public Operation {    
   public:
    Tensor weights;
    Tensor learningRate;
    WeightedOperation(Tensor* input);
    WeightedOperation();
    void updateWeights(Tensor& factor);
    virtual void gradWeights() {}
    virtual Tensor getGradWeights() { return Tensor(); }
};

class NeuralNetwork {
   public:
    std::vector<Operation> sequence;

    NeuralNetwork(std::vector<Operation> sequence);

    void backpropagate();
};

#endif