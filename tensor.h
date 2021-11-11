#include <memory>
#include <vector>

#ifndef TENSOR_H
#define TENSOR_H

/**
 * @brief Implementation of a mutable tensor. This class includes both static 
 * operations that allow the user to specify references to input and output tensors,
 * and non-static operations that do not mutate the tensor, instead returning
 * a new tensor. A tensor's data is stored in the form of an array, referenced
 * by a std::shared_ptr<float[]>. The default = operator will do a deep copy
 * of the tensor's shape, but NOT the array (contents) of a tensor. This
 * is due to the fact that a common tensor operation involves creating a new tensor
 * with a different shape than the previous tensor, but with the same contents.
 * To perform a full deep copy of a tensor, use the Tensor(const Tensor& t)
 * constructor.
 */
class Tensor {
   public:
    std::shared_ptr<float[]> array;
    std::vector<int> shape;
    int length;

    Tensor();
    Tensor(std::shared_ptr<float[]> array, std::vector<int>& shape);
    Tensor(std::vector<float>& array, std::vector<int>& shape);
    Tensor(float val, std::vector<int> shape);
    Tensor(const Tensor& t);
    Tensor(std::vector<int> shape);  //TODO: find out why this can't pass by reference

    void print();

    static void relu(const Tensor& t, Tensor& out);
    Tensor relu() const;

    static void softmax(const Tensor& t, Tensor& out);
    Tensor softmax() const;

    static void add(const Tensor& t, const Tensor& add, Tensor& out);
    Tensor add(const Tensor& other) const;

    static void scalarMult(const Tensor& t, float scalar, Tensor& out);
    Tensor scalarMult(float scalar);

    static void matmult(const Tensor& one, const Tensor& two, Tensor& out);
    static Tensor matmult(const Tensor& one, const Tensor& two);

    static void elementmult(const Tensor& one, const Tensor& two, Tensor& out);

    static void outer_product(const Tensor& one, const Tensor& two, Tensor& out);

    static float mse(const Tensor& exp, const Tensor& actual);

    static Tensor random(const std::vector<int>& shape);
};

#endif
