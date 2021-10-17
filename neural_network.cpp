#include "neural_network.h"

Layer::Layer(Tensor& matrix, Tensor& addVector, std::string& activatorType) {
    this->matrix = matrix;
    this->addVector = addVector;
    this->activatorType = activatorType;
}


NeuralNetwork::NeuralNetwork(std::vector<Layer> sequence) {
    this->sequence = sequence;
}

#include <iostream>
#include <memory>
using namespace std;

int main() {
    Tensor v1(shared_ptr<float[]>(new float[4]{1, 2, 3, 4}), {4, 1});
}