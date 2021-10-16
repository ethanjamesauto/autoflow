#include <memory>
#include <string>
#include "tensor.h"

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

struct Layer {
    Tensor matrix;
    Tensor addVector;
    Tensor input;
    Tensor output;
    Tensor gradient;
    std::string activatorType;
    Layer(Tensor& matrix, Tensor& addVector, std::string& activatorType);
};

class NeuralNetwork {
   public:
    std::shared_ptr<Layer[]> sequence;

    NeuralNetwork(std::shared_ptr<Layer[]> sequence);

    void backpropagate();
};

#endif