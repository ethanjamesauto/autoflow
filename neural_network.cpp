#include "neural_network.h"

Operation::Operation(std::string& operationType) {
    this->operationType = operationType;
}

NeuralNetwork::NeuralNetwork(std::vector<Operation> sequence) {
    this->sequence = sequence;
}

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
using namespace std;

Tensor gradMse(Tensor& exp, Tensor& actual) {
    assert(exp.shape == actual.shape);
    Tensor grad(0., exp.shape);
    float scalar = 2. / exp.length;
    for (int i = 0; i < exp.length; i++) {
        grad.array[i] = scalar * (exp.array[i] - actual.array[i]);
    }
    return grad;
}

Tensor gradSoftmax(Tensor& z) {
    Tensor ret(0., {z.length, z.length});
    float x = 0;
    for (int i = 0; i < z.length; i++) {
        x += exp(z.array[i]);
    }
    for (int k = 0; k < z.length; k++) {
        for (int l = 0; l < z.length; l++) {
            ret.array[k * z.length + l] = 0;
            if (k == l) {
                ret.array[k * z.length + l] += x * exp(z.array[k]);
            }
            ret.array[k * z.length + l] -= exp(z.array[k]) * exp(z.array[l]);
            ret.array[k * z.length + l] /= x * x;
        }
    }
    return ret;
}

float randFloat() {
    return rand() / (float)RAND_MAX;
}
int main() {
    int seed;
    cin >> seed;
    for (int i = 0; i < seed; i++) {
        randFloat();
    }
    for (int k = 0; k < 100; k++) {
        Tensor z({randFloat(), randFloat(), randFloat()}, {3, 1});
        float m;
        for (int epochs = 0; epochs < 1000; epochs++) {
            Tensor s = z.softmax();
            //cout << "Input: ";
            //s.print();
            Tensor a({.1, .7, .2}, {3, 1});
            m = Tensor::mse(s, a);

            Tensor gradM = gradMse(s, a);
            gradM.shape = {1, 3};
            Tensor gradSoft = gradSoftmax(z);
            Tensor gradient = Tensor::matmult(gradM, gradSoft);
            //gradient.print();

            gradient.shape = {3, 1};
            z.addMutable(gradient.scalarMult(-1.));
        }
        if (m > 1.e-5) {
            cout << "Error: " << m << endl;
        }
    }
}