#include "tensor.h"

#include <cstring>

int getArrayLength(std::vector<int> shape) {
    int ret = 1;
    for (int s : shape) {
        ret *= s;
    }
    return ret;
}

Tensor::Tensor(float* array, std::vector<int> shape) {
    this->array = array;
    this->shape = shape;
    this->length = getArrayLength(shape);
}

Tensor::Tensor(float val, const std::vector<int> shape) {
    this->shape = shape;
    this->length = getArrayLength(shape);
    this->array = new float[length];
    for (int i = 0; i < length; i++) {
        array[i] = val;
    }
}

Tensor::Tensor(const Tensor& t) {
    array = new float[t.length];
    memcpy(array, t.array, sizeof(float) * t.length);
    shape = t.shape;
    length = t.length;
}

Tensor::~Tensor() {
    delete[] array;
}

Tensor& Tensor::operator=(const Tensor& t) {
    if (this != &t) {
        float* newArr = new float[t.length];
        memcpy(newArr, t.array, sizeof(float) * t.length);

        delete[] array;

        array = newArr;
        shape = t.shape;
        length = t.length;
    }
    return *this;
}

#include <iostream>
using namespace std;

int main() {
    float* a = new float[9]{.1, -1, 0, 8, 2, .3, 7, 8, 6};
    float* b = new float[3]{7, 8, 2};
    vector<int> aV{3, 3};
    vector<int> bV{3, 1};
    Tensor tA(a, aV);
    Tensor tB(b, bV);
    Tensor tC = Tensor::matmult(tA, tB);
    float* d = new float[7]{1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0};
    vector<int> dV{1, 7};
    Tensor tD(d, dV);
    tD = tD.softmax();
    tD = tD.add(Tensor(50., tD.shape));
    for (int i = 0; i < tD.length; i++) {
        cout << tD.array[i] << " ";
    }
    cout << std::endl;
}
