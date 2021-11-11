#include "tensor.h"

#include <cstring>
#include <iostream>

#define SMALLER_FLOAT 3

/**
 * @brief Returns the length of the tensor's internal array using its shape.
 * 
 * @param shape The shape of the tensor
 * @return int The array length of the tensor
 */
int getArrayLength(std::vector<int> shape) {
    int ret = 1;
    for (int s : shape) {
        ret *= s;
    }
    return ret;
}

/**
 * @brief Construct a new Tensor::Tensor object with nothing initialized. This
 * does nothing, because it's often the case that an array of Tensor objects is
 * created without the desire to initialize those tensors until later.
 */
Tensor::Tensor() {}

/**
 * @brief Construct a new Tensor:: Tensor object with the desired data and shape.
 * 
 * @param array The data of the tensor (will be referenced by the tensor and
 * NOT copied)
 * @param shape The shape of the tensor (will be copied into the tensor)
 */
Tensor::Tensor(std::shared_ptr<float[]> array, std::vector<int>& shape) {
    this->array = array;
    this->shape = shape;
    this->length = getArrayLength(shape);
}

/**
 * @brief Construct a new Tensor::Tensor object. This constructor can be used
 * in one line in a manner similar to Tensor t({1, 2, 3, 4}, {4, 1}).
 * 
 * @param array The data of the tensor
 * @param shape The shape of the tensor
 */
Tensor::Tensor(std::vector<float>& array, std::vector<int>& shape) {
    this->shape = shape;
    this->length = getArrayLength(shape);
    this->array = std::shared_ptr<float[]>(new float[length]);
    for (int i = 0; i < length; i++) {
        this->array[i] = array[i];
    }
}

/**
 * @brief Construct a new Tensor::Tensor object with every value being initialized
 * to the desired value.
 * 
 * @param val The value to set every element of the tensor to
 * @param shape The shape of the tensor
 */
Tensor::Tensor(float val, const std::vector<int> shape) {
    this->shape = shape;
    this->length = getArrayLength(shape);
    this->array = std::shared_ptr<float[]>(new float[length]);
    for (int i = 0; i < length; i++) {
        array[i] = val;
    }
}

/**
 * @brief Construct a new Tensor::Tensor object that is a deep copy of another
 * tensor.
 * 
 * @param t The tensor to deep copy from
 */
Tensor::Tensor(const Tensor& t) {
    array = std::shared_ptr<float[]>(new float[t.length]);
    memcpy(array.get(), t.array.get(), sizeof(float) * t.length);
    shape = t.shape;
    length = t.length;
}

/**
 * @brief Construct a new Tensor::Tensor object.
 * The values of the tensor are NOT initialized.
 * 
 * @param shape The shape of the tensor
 */
Tensor::Tensor(std::vector<int> shape) {
    this->length = getArrayLength(shape);
    this->array = std::shared_ptr<float[]>(new float[length]);
    this->shape = shape;
}

/**
 * @brief Prints out this tensor to std::cout.
 * 
 */
void Tensor::print() {
    for (int i = 0; i < this->length; i++) {
        std::cout << this->array[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Generates a random float between (-1, 1) / SMALLER_FLOAT.
 * The SMALLER_FLOAT constant is used when large tensors can cause floats to
 * max out with NaN values.
 * 
 * @return float The random float
 */
inline float randFloat() {
    return (rand() / (float)RAND_MAX * 2 - 1) / SMALLER_FLOAT;
}

/**
 * @brief Returns a new random tensor with the given shape. The values will be
 * initialized with randFloat().
 * 
 * @param shape The shape of the tensor to create.
 * @return Tensor The random tensor
 */
Tensor Tensor::random(const std::vector<int>& shape) {
    Tensor t(shape);
    for (int i = 0; i < t.length; i++) {
        t.array[i] = randFloat();
    }
    return t;
}