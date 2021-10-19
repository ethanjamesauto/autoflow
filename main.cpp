#include "neural_network.h"
#include "operators.h"
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

void gradSoftmax(Tensor& z, Tensor& ret) {
    float x = 0;
    for (int i = 0; i < z.length; i++) {
        x += exp(z.array[i]);
    }
    for (int k = 0; k < z.length; k++) {
        for (int l = 0; l < z.length; l++) {
            float& val = ret.array[k * z.length + l];
            val = 0;
            if (k == l) {
                val += x * exp(z.array[k]);
            }
            val -= exp(z.array[k]) * exp(z.array[l]);
            val /= x * x;
        }
    }
}

float randFloat() {
    return rand() / (float)RAND_MAX * 2 - 1;
}

int size = 10;
int main() {
    int seed;
    cin >> seed;
    for (int i = 0; i < seed; i++) {
        randFloat();
    }
    for (int k = 0; k < 5; k++) {
        Tensor z(0., {size, 1});
        Tensor s;
        Tensor a(shared_ptr<float[]>(new float[size]), {size, 1});
        Tensor gradSoft(0., {z.length, z.length});
        for (int i = 0; i < z.length; i++) {
            z.array[i] = randFloat();
        }
        for (int i = 0; i < a.length; i++) {
            a.array[i] = randFloat();
        }
        cout << "Initial: ";
        z.print();
        a.softmaxMutable();
        cout << "Expected: ";
        a.print();
        Operation* mse = new MSE(&s, a);
        for (int epochs = 0; epochs < 1000; epochs++) {
            s = z.softmax();
            mse->input = &s;
            mse->execute();
            mse->gradOp();
            Tensor gradM = mse->gradWeight;
            gradM.shape = {1, size};
            gradSoftmax(z, gradSoft);
            Tensor gradient = Tensor::matmult(gradM, gradSoft);
            //gradient.print();
            gradient.shape = {size, 1};
            z.addMutable(gradient.scalarMult(-10.));
        }
        cout << "Actual: ";
        s.print();
        cout << "Error: " << mse->output.array[0] << endl;
        cout << endl;
    }
}