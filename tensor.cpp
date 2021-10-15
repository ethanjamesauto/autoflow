#include "tensor.h"

#include <cstring>

int getArrayLength(std::vector<int> shape) {
    int ret = 1;
    for (int s : shape) {
        ret *= s;
    }
    return ret;
}

Tensor::Tensor(std::shared_ptr<float[]> array, std::vector<int> shape) {
    this->array = array;
    this->shape = shape;
    this->length = getArrayLength(shape);
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

#include <iostream>
using namespace std;

int main() {
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
}
