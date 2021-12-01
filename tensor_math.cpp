#include "tensor.h"

#include <cassert>
#include <cmath>

/**
 * @brief Returns the max of the two given float values.
 * 
 * @param a The first input
 * @param b The second input
 * @return constexpr float The max of the two inputs
 */
constexpr float max(const float a, const float b) {
    return a > b ? a : b;
}

/**
 * @brief Applies the ReLU (Rectified Linear Unit) function to a tensor
 * 
 * @param t The tensor to use as the input of the function. It is not modified.
 * @param out The tensor to store the output of the function.
 */
void Tensor::relu(const Tensor& t, Tensor& out) {
    for (int i = 0; i < t.length; i++) {
        out.array[i] = max(t.array[i], 0);
    }
}

/**
 * @brief Applies the ReLU function to the tensor and returns the result.
 * 
 * @return Tensor The tensor with the ReLu function applied.
 */
Tensor Tensor::relu() const {
    Tensor t(shape);
    relu(*this, t);
    return t;
}

/**
 * @brief Takes two tensor inputs of equal shape, adds them together, and stores
 * the result.
 * 
 * @param t The first tensor to be added. It is not modified.
 * @param other The second tensor to be added. It is not modified.
 * @param out The tensor to store the result of the add operation
 */
void Tensor::add(const Tensor& t, const Tensor& other, Tensor& out) {
    assert(t.shape == other.shape);
    for (int i = 0; i < t.length; i++) {
        out.array[i] = t.array[i] + other.array[i];
    }
}

/**
 * @brief Adds another tensor to this tensor and returns the result.
 * 
 * @param other The tensor to add to this tensor
 * @return Tensor The result of the add operation
 */
Tensor Tensor::add(const Tensor& other) const {
    Tensor t(other.shape);
    add(*this, other, t);
    return t;
}

/**
 * @brief Multiplies a tensor by a scalar, and stores the result.
 * 
 * @param t The tensor to be multiplied. It is not modified.
 * @param scalar The scalar to multiply the tensor by
 * @param out The tensor to store the result of the operation
 */
void Tensor::scalarMult(const Tensor& t, float scalar, Tensor& out) {
    for (int i = 0; i < t.length; i++) {
        out.array[i] = t.array[i] * scalar;
    }
}

/**
 * @brief Multiplies this tensor by a scalar, and returns the result
 * 
 * @param scalar The scalar to multiply this tensor by
 * @return Tensor The tensor resulting from the operation
 */
Tensor Tensor::scalarMult(float scalar) {
    Tensor t(shape);
    scalarMult(*this, scalar, t);
    return t;
}

/**
 * @brief Applies the softmax function to a tensor, and stores the result.
 * 
 * @param t The tensor to apply the softmax function to
 * @param out The tensor to store the result of the softmax function
 */
void Tensor::softmax(const Tensor& t, Tensor& out) {
    double sum;
    for (int i = 0; i < t.length; i++) {
        out.array[i] = exp(t.array[i]);
        sum += out.array[i];
    }
    for (int i = 0; i < t.length; i++) {
        out.array[i] /= sum;
    }
}

/**
 * @brief Applies the softmax function to this tensor, and returns the result
 * 
 * @return Tensor This tensor with the softmax function applied
 */
Tensor Tensor::softmax() const {
    Tensor t(shape);
    softmax(*this, t);
    return t;
}

/**
 * @brief Returns the shape of the resulting tensor created by multiplying
 * two tensors.
 * 
 * @param a The shape of the first tensor
 * @param b The shape of the second tensor
 * @return std::vector<int> The resulting shape
 */
std::vector<int> resultShape(const std::vector<int>& a, const std::vector<int>& b) {
    assert(a[1] == b[0]);
    std::vector<int> ret{a[0], b[1]};
    return ret;
}

/**
 * @brief Multiplies two tensors together, and stores the result.
 * 
 * @param one The first tensor to be multiplied
 * @param two The second tensor to be multiplied
 * @param out The tensor to store the result of the multiplication
 */
void Tensor::matmult(const Tensor& one, const Tensor& two, Tensor& out) {
    assert(out.shape == resultShape(one.shape, two.shape));
    int row1 = one.shape[0];
    int row2 = two.shape[0];
    int col1 = one.shape[1];
    int col2 = two.shape[1];
    for (int r = 0; r < row1; r++) {
        for (int ansC = 0; ansC < col2; ansC++) {
            float dot = 0;
            for (int c = 0; c < col1; c++) {
                dot += one.array[r * col1 + c] * two.array[c * col2 + ansC];
            }
            out.array[r * col2 + ansC] = dot;
        }
    }
}

/**
 * @brief Multiplies two tensors together, and returns the result. 
 * 
 * @param one The first tensor to be multiplied
 * @param two The second tensor to be multiplied
 * @return Tensor The result of the multiplication operation
 */
Tensor Tensor::matmult(const Tensor& one, const Tensor& two) {
    Tensor result = Tensor(0., resultShape(one.shape, two.shape));
    matmult(one, two, result);
    return result;
}

/**
 * @brief Performs element-wise multiplication of two tensors, and stores the
 * result.
 * @param one The first tensor to be multiplied
 * @param two The second tensor to be multiplied
 * @param out The tensor to store the result of the operation
 */
void Tensor::elementmult(const Tensor& one, const Tensor& two, Tensor& out) {
    assert(one.shape == two.shape && one.shape == out.shape);
    for (int i = 0; i < one.length; i++) {
        out.array[i] = one.array[i] * two.array[i];
    }
}

/**
 * @brief Computes the outer product of two tensors, and stores the result.
 * 
 * @param one The first tensor to be multiplied
 * @param two The second tensor to be multiplied
 * @param out The tensor to store the result of the operation
 */
void Tensor::outer_product(const Tensor& one, const Tensor& two, Tensor& out) {
    assert(one.shape.size() == 2 && two.shape.size() == 2);
    assert(one.shape[1] == 1 && two.shape[1] == 1);
    assert(out.shape[0] == one.shape[0] && out.shape[1] == two.shape[0]);
    for (int r = 0; r < out.shape[0]; r++) {
        for (int c = 0; c < out.shape[1]; c++) {
            out.array[r * out.shape[1] + c] = one.array[r] * two.array[c];
        }
    }
}

/**
 * @brief Computes the mean squared error (MSE) between two tensors
 * 
 * @param exp The tensor storing the experimental value
 * @param actual The tensor storing the actual value
 * @return float The mean squared error
 */
float Tensor::mse(const Tensor& exp, const Tensor& actual) {
    assert(exp.shape == actual.shape);
    float ans = 0;
    for (int i = 0; i < exp.length; i++) {
        ans += std::pow(actual.array[i] - exp.array[i], 2.);
    }
    return ans / exp.length;
}