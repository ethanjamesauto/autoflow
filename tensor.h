#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>

/**
 * @brief Implementation of a tensor with variable shape. Essentially equivalent to a TensorFlow tensor.
 */
class Tensor {
   public:
    std::shared_ptr<float[]> array;
    std::vector<int> shape;
    int length;

    Tensor();
    Tensor(std::shared_ptr<float[]> array, std::vector<int> shape);
    Tensor(float val, std::vector<int> shape);
    Tensor(const Tensor& t);

    void reluMutable();
    void softmaxMutable();
    void addMutable(const Tensor& other);

    Tensor relu() const;
    Tensor softmax() const;
    Tensor add(const Tensor& other) const;

    Tensor reshape(std::vector<int> shape) const;

    static void matmult(const Tensor& one, const Tensor& two, Tensor& result_container);
    static Tensor matmult(const Tensor& one, const Tensor& two);
};

#endif
