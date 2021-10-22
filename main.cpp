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

int size = 10;
int main() {
    srand(time(NULL));
    for (int k = 0; k < 10; k++) {
        Tensor z({size, 1});
        Tensor a({size, 1});
        Tensor w({size, size});
        Tensor w1({size, 1});
        Tensor e({size, size});
        Tensor e1({size, 1});

        for (int i = 0; i < z.length; i++) {
            z.array[i] = randFloat();
            a.array[i] = randFloat();
            for (int j = 0; j < z.length; j++) {
                w.array[i * z.length + j] = randFloat();
                e.array[i * z.length + j] = randFloat();
            }
            w1.array[i] = randFloat();
            e1.array[i] = randFloat();
        }

        Tensor::softmax(a, a);
        //mult->add->relu->mult2->add2->soft->mse
        //gradients:
        //nxn ->nx1->nx1 -> nxn ->nx1 ->nxn ->nx1
        MatrixMult* mult = new MatrixMult(&z, w);
        MatrixAdd* add = new MatrixAdd(&mult->output, w1);
        Operation* relu = new Relu(&add->output);

        MatrixMult* mult2 = new MatrixMult(&relu->output, e);
        MatrixAdd* add2 = new MatrixAdd(&mult2->output, e1);
        Operation* soft = new Softmax(&add2->output);
        Operation* mse = new MSE(&soft->output, a);

        for (int epochs = 0; epochs < 1000; epochs++) {
            mult->execute();
            add->execute();
            relu->execute();

            mult2->execute();
            add2->execute();
            soft->execute();
            mse->execute();

            if (epochs == 0) {
                cout << "Initial: ";
                soft->output.print();
            }

            mult->gradOp();
            add->gradOp();
            relu->gradOp();

            mult2->gradOp();
            add2->gradOp();
            soft->gradOp();
            mse->gradOp();

            Tensor mseT = mse->getGradOp();
            mseT.shape = {1, mseT.length};
            Tensor mseSoft = Tensor::matmult(mseT, soft->getGradOp());
            Tensor mseMult2({1, mseSoft.length});
            Tensor::matmult(mseSoft, mult2->getGradOp(), mseMult2);
            mseMult2.shape = {mseMult2.length, 1};
            Tensor mseRelu(mseMult2.shape);
            Tensor::elementmult(mseMult2, relu->getGradOp(), mseRelu);
        
            mseSoft.shape = {mseSoft.length, 1};
            Tensor::add(add2->weights, mseSoft.scalarMult(-10.), add2->weights);
            
            Tensor mult2W({mseSoft.length, mseSoft.length});
            Tensor::outer_product(mseSoft, mult2->getGradWeights(), mult2W);
            Tensor::add(mult2->weights, mult2W.scalarMult(-10.), mult2->weights);

            Tensor::add(add->weights, mseRelu.scalarMult(-10.), add->weights);

            Tensor multW({mseRelu.length, mseRelu.length});
            Tensor::outer_product(mseRelu, mult->getGradWeights(), multW);
            Tensor::add(mult->weights, multW.scalarMult(-10.), mult->weights);

            //Tensor::add(mult->weights, gradW.scalarMult(-10.), mult->weights);
        }
        cout << "Expected: ";
        a.print();
        cout << "Experimental: ";
        soft->output.print();
        cout << "Error: " << mse->output.array[0] << endl;
        cout << endl;
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