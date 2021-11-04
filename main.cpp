#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "neural_network.h"
#include "operators.h"
using namespace std;

float randFloat1() {
    return rand() / (float)(RAND_MAX + 1);
}

int isize = 30;
int osize = 3;
int numFeatures = 2;
int numEpochs = 1500;
int numLayers = 6;

int main() {
    srand(time(NULL));
    srand(5);
    Tensor features[numFeatures];
    Tensor labels[numFeatures];
    for (int i = 0; i < numFeatures; i++) {
        features[i] = Tensor::random({isize, 1});
        labels[i] = Tensor::random({osize, 1});
    }

    Operation* operations[numLayers];

    //mult->add->relu->mult2->add2->soft->mse
    //gradients:
    //nxn ->nx1->nx1 -> nxn ->nx1 ->nxn ->nx1
    MatrixMult mult = MatrixMult(&features[0], isize);
    MatrixAdd add = MatrixAdd(&mult.output);
    Sigmoid relu = Sigmoid(&add.output);
    MatrixMult mult2 = MatrixMult(&relu.output, osize);
    MatrixAdd add2 = MatrixAdd(&mult2.output);
    MSE mse = MSE(&add2.output, &labels[0]);

    operations[0] = &mult;
    operations[1] = &add;
    operations[2] = &relu;
    operations[3] = &mult2;
    operations[4] = &add2;
    operations[5] = &mse;

    for (int epochs = 0; epochs < numEpochs; epochs++) {
        int index = (int)(randFloat1() * numFeatures);
        mult.input = &features[index];
        mse.actual = &labels[index];

        for (int i = 0; i < numLayers; i++) {
            operations[i]->execute();
            operations[i]->gradOp();
        }
        Tensor mseT = mse.getGradOp();
        add2.RMSProp(mseT);

        Tensor mult2W(mult2.weights.shape);
        Tensor::outer_product(mseT, mult2.getGradWeights(), mult2W);
        mult2.RMSProp(mult2W);

        mseT.shape = {1, mse.input->length};
        Tensor mse_mult({1, mult2.input->length});
        Tensor::matmult(mseT, mult2.getGradOp(), mse_mult);
        mse_mult.shape = {mse_mult.length, 1};

        Tensor::elementmult(mse_mult, relu.getGradOp(), mse_mult);
        add.RMSProp(mse_mult);

        Tensor multW({mse_mult.length, mse_mult.length});
        Tensor::outer_product(mse_mult, mult.getGradWeights(), multW);
        mult.RMSProp(multW);
    }
    for (int i = 0; i < numFeatures; i++) {
        int index = i;
        operations[0]->input = &features[index];
        mse.actual = &labels[index];

        for (int i = 0; i < numLayers; i++) {
            operations[i]->execute();
        }
        cout << "Input: ";
        operations[0]->input->print();
        cout << "Expected: ";
        mse.actual->print();
        cout << "Experimental: ";
        add2.output.print();
        cout << "Error: ";
        mse.output.print();
        cout << endl;
    }
    /*
    cout << "Weights:" << endl;
    mult.weights.print();
    add.weights.print();
    mult2.weights.print();
    add2.weights.print();
    */
    return 0;
}
