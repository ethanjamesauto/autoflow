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

Tensor::Tensor() {}

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

/**
 * @brief Construct a new Tensor:: Tensor object.
 * The values of the tensor are NOT initialized.
 * 
 * @param shape shape of the Tensor
 */
Tensor::Tensor(std::vector<int> shape) {
    this->length = getArrayLength(shape);
    this->array = std::shared_ptr<float[]>(new float[length]);
    this->shape = shape;
}

void Tensor::print() {
    for (int i = 0; i < this->length; i++) {
        std::cout << this->array[i] << " ";
    }
    std::cout << std::endl;
}
