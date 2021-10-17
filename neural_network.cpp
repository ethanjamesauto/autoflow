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
    Tensor v2(shared_ptr<float[]>(new float[3]{5, 6, 7}), {3, 1});
    Tensor out(0., {4, 3});
    Tensor::outer_product(v1, v2, out);
    for (int i = 0; i < out.length; i++) {
        cout << out.array[i] << " ";
    }
    cout << std::endl;
}