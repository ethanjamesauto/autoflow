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
    Tensor in({1, 2}, {2, 1});
    Tensor w({1, 2, 3, 4}, {2, 2});
    Tensor out = Tensor::matmult(w, in);
    Tensor actual({1, 2}, {2, 1});
    for (int i = 0; i < out.length; i++) {
        cout << out.array[i] << " ";
    }
    cout << std::endl;
}