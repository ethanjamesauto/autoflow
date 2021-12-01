#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "neural_network.h"
#include "operators.h"
using namespace std;

float randFloat1() {
    return rand() / (float)(RAND_MAX + 1);
}

int isize = 28 * 28;
int osize = 10;
int numFeatures = 18000;
int numTesting = 1000;
int numEpochs = 2000;
int numLayers = 7;
int batch_size = 32;

int main() {
    srand(time(NULL));  //set up random number generator to the time
    srand(36);          //use a pre-determined seed; comment out this line for different rng for each run

    Operation* operations[numLayers];
    Tensor features[numFeatures + numTesting];
    Tensor labels[numFeatures + numTesting];

    //read handwritten numbers and their values into data.
    //note: this is very slow
    ifstream f("data.txt");
    for (int i = 0; i < numFeatures + numTesting; i++) {
        if (i % 100 == 0) {
            cout << "Reading from file, line: " << i << endl;
        }
        features[i] = Tensor({isize, 1});
        labels[i] = Tensor({osize, 1});
        string s;
        getline(f, s);
        stringstream ss(s);
        for (int j = 0; j < 28 * 28; j++) {
            ss >> features[i].array[j];
        }
        getline(f, s);
        ss = stringstream(s);
        for (int j = 0; j < 10; j++) {
            ss >> labels[i].array[j];
        }
    }

    //set up each individual layer, then add them into an array for easy execution
    MatrixMult mult = MatrixMult(&features[0], 512);
    MatrixAdd add = MatrixAdd(&mult.output);
    Relu relu = Relu(&add.output);
    MatrixMult mult2 = MatrixMult(&relu.output, osize);
    MatrixAdd add2 = MatrixAdd(&mult2.output);
    Softmax soft = Softmax(&add2.output);
    CategoricalCrossEntropy cce = CategoricalCrossEntropy(&soft.output, &labels[0]);
    operations[0] = &mult;
    operations[1] = &add;
    operations[2] = &relu;
    operations[3] = &mult2;
    operations[4] = &add2;
    operations[5] = &soft;
    operations[6] = &cce;

    for (int epochs = 0; epochs < numEpochs; epochs++) {
        cout << "Epoch: " << epochs << " of " << numEpochs << endl;

        //set up tensors to store weights for mini-batch gradient descent
        Tensor add2G(0, add2.weights.shape);
        Tensor mult2G(0, mult2.weights.shape);
        Tensor addG(0, add.weights.shape);
        Tensor multG(0, mult.weights.shape);

        for (int i = 0; i < batch_size; i++) {
            //set feature and label
            int index = (int)(randFloat1() * numFeatures);
            mult.input = &features[index];
            cce.actual = &labels[index];

            //execute the model, and calculate the gradient
            for (int i = 0; i < numLayers; i++) {
                operations[i]->execute();
                operations[i]->gradOp();
            }

            //backpropagate
            Tensor mseT = cce.getGradOp();
            Tensor::add(add2G, mseT, add2G);

            Tensor mult2W(mult2.weights.shape);
            Tensor::outer_product(mseT, mult2.getGradWeights(), mult2W);
            Tensor::add(mult2G, mult2W, mult2G);

            mseT.shape = {1, cce.input->length};
            Tensor mse_mult({1, mult2.input->length});
            Tensor::matmult(mseT, mult2.getGradOp(), mse_mult);
            mse_mult.shape = {mse_mult.length, 1};

            Tensor::elementmult(mse_mult, relu.getGradOp(), mse_mult);
            Tensor::add(addG, mse_mult, addG);

            Tensor multWeights = mult.getGradWeights();
            Tensor multW({mse_mult.length, multWeights.length});
            Tensor::outer_product(mse_mult, multWeights, multW);
            Tensor::add(multG, multW, multG);
        }

        //update weights
        Tensor::scalarMult(add2G, 1. / batch_size, add2G);
        add2.RMSProp(add2G);

        Tensor::scalarMult(mult2G, 1. / batch_size, mult2G);
        mult2.RMSProp(mult2G);

        Tensor::scalarMult(addG, 1. / batch_size, addG);
        add.RMSProp(addG);

        Tensor::scalarMult(multG, 1. / batch_size, multG);
        mult.RMSProp(multG);
    }

    //run testing data and see how accurate the model is
    int numSuccessful = 0;
    for (int i = numFeatures; i < numFeatures + numTesting; i++) {
        mult.input = &features[i];
        cce.actual = &labels[i];

        for (int i = 0; i < numLayers; i++) {
            operations[i]->execute();
        }
        for (int i = 0; i < soft.output.length; i++) {
            if (abs(cce.actual->array[i] - 1.) < 0.0001 && soft.output.array[i] > .5) {
                numSuccessful++;
                break;
            }
        }
        //cout << "Input: ";
        //mult.input->print();
        /*
        cout << "Expected: ";
        cce.actual->print();
        cout << "Experimental: ";
        soft.output.print();
        cout << "Error: ";
        cce.output.print();
        cout << endl;
        //*/
    }
    for (int i = 0; i < numLayers; i++) {
        cout << "Layer " << i + 1 << " input and output sizes: (" << operations[i]->input->shape[0] << ", " << operations[i]->output.shape[0] << ")" << endl;
    }
    cout << "Accuracy: " << (float)numSuccessful / numTesting << endl;
    /*
    cout << "Weights:" << endl;
    mult.weights.print();
    add.weights.print();
    mult2.weights.print();
    add2.weights.print();
    //*/
}