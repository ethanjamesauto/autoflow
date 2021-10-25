#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "neural_network.h"
#include "operators.h"
using namespace std;

float randFloat() {
    return rand() / (float)RAND_MAX * 2 - 1;
}

int isize = 10;
int numFeatures = 7;
int numEpochs = 10000;
float learning_rate = .23;
int main() {
    srand(time(NULL));
    srand(9);
    for (int k = 0; k < 1; k++) {
        Tensor z[numFeatures];
        Tensor a[numFeatures];
        Tensor w({isize, isize});
        Tensor w1({isize, 1});
        Tensor e({isize, isize});
        Tensor e1({isize, 1});
        for (int i = 0; i < numFeatures; i++) {
            z[i] = Tensor({isize, 1});
            a[i] = Tensor({isize, 1});
            for (int j = 0; j < isize; j++) {
                z[i].array[j] = randFloat();
                a[i].array[j] = randFloat();
            }
        }
        for (int i = 0; i < isize; i++) {
            for (int j = 0; j < isize; j++) {
                w.array[i * isize + j] = randFloat();
                e.array[i * isize + j] = randFloat();
            }
            w1.array[i] = randFloat();
            e1.array[i] = randFloat();
        }

        //mult->add->relu->mult2->add2->soft->mse
        //gradients:
        //nxn ->nx1->nx1 -> nxn ->nx1 ->nxn ->nx1
        MatrixMult* mult = new MatrixMult(&z[0], w);
        MatrixAdd* add = new MatrixAdd(&mult->output, w1);
        //MatrixMult* mult2 = new MatrixMult(&add->output, e);
        Relu* relu = new Relu(&add->output);
        MatrixMult* mult2 = new MatrixMult(&relu->output, e);
        MatrixAdd* add2 = new MatrixAdd(&mult2->output, e1);
        MSE* mse = new MSE(&add2->output, &a[0]);
        for (int epochs = 0; epochs < numEpochs; epochs++) {
            mult->input = &z[epochs % numFeatures];
            mse->actual = &a[epochs % numFeatures];

            mult->execute();
            add->execute();
            relu->execute();
            mult2->execute();
            add2->execute();
            mse->execute();

            if (epochs == 0) {
                cout << "Initial: ";
                add2->output.print();
            }

            mult->gradOp();
            add->gradOp();
            relu->gradOp();
            mult2->gradOp();
            add2->gradOp();
            mse->gradOp();

            Tensor mseT = mse->getGradOp();

            Tensor::add(add2->weights, mseT.scalarMult(-learning_rate), add2->weights);

            Tensor mult2W({mseT.length, mseT.length});
            Tensor::outer_product(mseT, mult2->getGradWeights(), mult2W);
            Tensor::add(mult2->weights, mult2W.scalarMult(-learning_rate), mult2->weights);

            mseT.shape = {1, mseT.length};
            Tensor mse_mult({mseT.shape});
            Tensor::matmult(mseT, mult2->getGradOp(), mse_mult);
            mse_mult.shape = {mse_mult.length, 1};

            Tensor::elementmult(mse_mult, relu->getGradOp(), mse_mult);

            Tensor::add(add->weights, mse_mult.scalarMult(-learning_rate), add->weights);

            Tensor multW({mse_mult.length, mse_mult.length});
            Tensor::outer_product(mse_mult, mult->getGradWeights(), multW);
            Tensor::add(mult->weights, multW.scalarMult(-learning_rate), mult->weights);

            //if (add2->output.array[0] != add2->output.array[0])
            //    cout << "poop";

            if (numEpochs - epochs <= numFeatures) {
                cout << "Input: ";
                mult->input->print();
                cout << "Expected: ";
                mse->actual->print();
                cout << "Experimental: ";
                add2->output.print();
                cout << "Error: " << mse->output.array[0] << endl;
                cout << endl;
            }
        }
    }
    return 0;
}

int main2() {
    Tensor w({1., 2, 3, 4}, {2, 2});
    Tensor a({5., 6.}, {2, 1});
    Operation* mult = new MatrixMult(&a, Tensor(w));
    mult->execute();
    mult->gradOp();
    mult->output.print();
    mult->getGradWeights().print();
    mult->getGradOp().print();
    w = w.add(Tensor({-2, -2, -2, -2}, {2, 2}));
    Relu rel(&w);
    rel.execute();
    rel.gradOp();
    rel.getGradOp().print();
    w.print();
    return 0;
}