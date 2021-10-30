#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "neural_network.h"
#include "operators.h"
using namespace std;

float randFloat1() {
    return rand() / (float)RAND_MAX;
}

int isize = 10;
int numFeatures = 5;
int numEpochs = 10000;
float learning_rate = .15;

int main() {
    srand(time(NULL));
    srand(5);
    Tensor z[numFeatures];
    Tensor a[numFeatures];
    for (int i = 0; i < numFeatures; i++) {
        z[i] = Tensor::random({isize, 1});
        a[i] = Tensor::random({isize, 1});
    }

    Operation* operations[6];

    //mult->add->relu->mult2->add2->soft->mse
    //gradients:
    //nxn ->nx1->nx1 -> nxn ->nx1 ->nxn ->nx1
    MatrixMult mult = MatrixMult(&z[0], Tensor::random({isize, isize}));
    MatrixAdd add = MatrixAdd(&mult.output, Tensor::random({isize, 1}));
    Sigmoid relu = Sigmoid(&add.output);
    MatrixMult mult2 = MatrixMult(&relu.output, Tensor::random({isize, isize}));
    MatrixAdd add2 = MatrixAdd(&mult2.output, Tensor::random({isize, 1}));
    MSE mse = MSE(&add2.output, &a[0]);

    mult.learningRate = learning_rate;
    add.learningRate = learning_rate;
    mult2.learningRate = learning_rate;
    add2.learningRate = learning_rate;

    operations[0] = &mult;
    operations[1] = &add;
    operations[2] = &relu;
    operations[3] = &mult2;
    operations[4] = &add2;
    operations[5] = &mse;

    for (int epochs = 0; epochs < numEpochs; epochs++) {
        int index = (int)(randFloat1() * numFeatures);
        mult.input = &z[index];
        mse.actual = &a[index];

        for (int i = 0; i < 6; i++) {
            operations[i]->execute();
            operations[i]->gradOp();
        }

        Tensor mseT = mse.getGradOp();

        Tensor::add(add2.weights, mseT.scalarMult(-add2.learningRate), add2.weights);

        Tensor mult2W({mseT.length, mseT.length});
        Tensor::outer_product(mseT, mult2.getGradWeights(), mult2W);
        Tensor::add(mult2.weights, mult2W.scalarMult(-mult2.learningRate), mult2.weights);

        mseT.shape = {1, mseT.length};
        Tensor mse_mult({mseT.shape});
        Tensor::matmult(mseT, mult2.getGradOp(), mse_mult);
        mse_mult.shape = {mse_mult.length, 1};

        Tensor::elementmult(mse_mult, relu.getGradOp(), mse_mult);

        Tensor::add(add.weights, mse_mult.scalarMult(-add.learningRate), add.weights);

        Tensor multW({mse_mult.length, mse_mult.length});
        Tensor::outer_product(mse_mult, mult.getGradWeights(), multW);
        Tensor::add(mult.weights, multW.scalarMult(-mult.learningRate), mult.weights);

        if (numEpochs - epochs <= numFeatures) {
            //cout << "Input: ";
            //mult.input->print();
            //cout << "Expected: ";
            //mse.actual->print();
            //cout << "Experimental: ";
            //add2.output.print();
            cout << "Error: " << mse.output.array[0] << endl;
            //cout << endl;
        }
    }
    return 0;
}
