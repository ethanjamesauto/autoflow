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
#include <cassert>
using namespace std;

Tensor gradMse(Tensor& exp, Tensor& actual)  {
    assert(exp.shape == actual.shape);
    Tensor grad(0., exp.shape);
    float scalar = 2. / exp.length;
    for (int i = 0; i < exp.length; i++) {
        grad.array[i] = scalar * (exp.array[i] - actual.array[i]);
    }
    return grad;
}

int main() {
    Tensor in({1, 2}, {2, 1});
    Tensor w({1, 2, 3, 4}, {2, 2});
    Tensor out = Tensor::matmult(w, in);
    Tensor actual({7, 2}, {2, 1});
    Tensor grad = gradMse(out, actual);
    for (int i = 0; i < out.length; i++) {
        cout << grad.array[i] << " ";
    }
    cout << std::endl << Tensor::mse(out, actual);
}