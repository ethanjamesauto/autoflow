#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

/**
 * @brief Implementation of a tensor with variable shape. Essentially equivalent to a TensorFlow tensor.
 */
class Tensor {
   public:
    float* array;
    std::vector<int> shape;
    int length;

    Tensor(float* array, std::vector<int> shape);
    Tensor(float val, std::vector<int> shape);
    Tensor(const Tensor& t);
    ~Tensor();
    Tensor& operator=(const Tensor& t);

    void reluMutable();
    void softmaxMutable();
    void addMutable(const Tensor& other);

    Tensor reshape(std::vector<int> shape) const;
    Tensor relu() const;
    Tensor softmax() const;
    static Tensor matmult(const Tensor& one, const Tensor& two);
    static void matmult(const Tensor& one, const Tensor& two, Tensor& result_container);
};

#endif
