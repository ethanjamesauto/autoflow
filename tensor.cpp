#include "tensor.h"

#include <cstring>
#include <iostream>

int getArrayLength(std::vector<int> shape) {
    int ret = 1;
    for (int s : shape) {
        ret *= s;
    }
    return ret;
}

Tensor::Tensor() {
    this->array = std::shared_ptr<float[]>(new float[0]);
    this->shape = {};
    this->length = 0;
}

Tensor::Tensor(std::shared_ptr<float[]> array, std::vector<int> shape) {
    this->array = array;
    this->shape = shape;
    this->length = getArrayLength(shape);
}

Tensor::Tensor(std::vector<float> array, std::vector<int> shape) {
    this->shape = shape;
    this->length = getArrayLength(shape);
    this->array = std::shared_ptr<float[]>(new float[length]);
    for (int i = 0; i < length; i++) {
        this->array[i] = array[i];
    }
}

Tensor::Tensor(float val, const std::vector<int> shape) {
    this->shape = shape;
    this->length = getArrayLength(shape);
    this->array = std::shared_ptr<float[]>(new float[length]);
    for (int i = 0; i < length; i++) {
        array[i] = val;
    }
}

Tensor::Tensor(const Tensor& t) {
    array = std::shared_ptr<float[]>(new float[t.length]);
    memcpy(array.get(), t.array.get(), sizeof(float) * t.length);
    shape = t.shape;
    length = t.length;
}

void Tensor::print() {
    for (int i = 0; i < this->length; i++) {
        std::cout << this->array[i] << " ";
    }
    std::cout << std::endl;
}

#include <iostream>
using namespace std;

int main1() {
    Tensor tA(shared_ptr<float[]>(new float[9]{.1, -1, 0, 8, 2, .3, 7, 8, 6}), {3, 3});
    Tensor tB(shared_ptr<float[]>(new float[3]{7, 8, 2}), {3, 1});
    Tensor tC = Tensor::matmult(tA, tB);
    shared_ptr<float[]> d(new float[7]{1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0});
    vector<int> dV{1, 7};
    Tensor tD(d, dV);
    tD = tD.softmax();
    tD = tD.add(Tensor(50., tD.shape));
    for (int i = 0; i < tD.length; i++) {
        cout << tD.array[i] << " ";
    }
    cout << std::endl;
    return 0;
}
