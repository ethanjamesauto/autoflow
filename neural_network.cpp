#include "neural_network.h"

Layer::Layer(Tensor& matrix, Tensor& addVector, std::string activatorType) {
    this->matrix = matrix;
    this->addVector = addVector;
    this->activatorType = activatorType;
}

NeuralNetwork::NeuralNetwork(std::shared_ptr<Layer[]> sequence) {
    this->sequence = sequence;
}
