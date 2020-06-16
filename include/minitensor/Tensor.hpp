#ifndef MINITENSOR_TENSOR_HPP
#define MINITENSOR_TENSOR_HPP
#include "defines.hpp"

#include "Shape.hpp"
#include "utilities.hpp"

#include <cstddef>

namespace mt
{
    template <class T, uint8_t D>
    class TensorIterator;

    template <class T, uint8_t D, class ENABLE = void>
    class Tensor;

    // This class handles creating the operator []
    template <class DERIVED, class DTYPE, uint8_t D>
    class ConstTensorIndexing
    {
      public:
        Tensor<const DTYPE, D - 1> operator[](uint32_t i) const
        {
            const Shape<D>& shape = static_cast<const DERIVED*>(this)->getShape();
            const DTYPE* ptr = static_cast<const DERIVED*>(this)->getData();
            ptr += shape.getStride(0) * i;
            Shape<D - 1> out_shape = stripOuterDim(shape);
            return Tensor<const DTYPE, D - 1>(ptr, std::move(out_shape));
        }
        template <class... ARGS>
        const DTYPE& operator()(ARGS&&... args) const
        {
            return *static_cast<const DERIVED*>(this)->ptr(std::forward<ARGS>(args)...);
        }
    };

    template <class DERIVED, class DTYPE>
    class ConstTensorIndexing<DERIVED, DTYPE, 1>
    {
      public:
        const DTYPE& operator[](uint32_t i) const { return *static_cast<const DERIVED*>(this)->ptr(i); }
        template <class... ARGS>
        const DTYPE& operator()(ARGS&&... args) const
        {
            return *static_cast<const DERIVED*>(this)->ptr(std::forward<ARGS>(args)...);
        }
    };

    template <class DERIVED, class DTYPE, uint8_t D, class Enable = void>
    class TensorIndexing : public ConstTensorIndexing<DERIVED, DTYPE, D>
    {
      public:
        Tensor<DTYPE, D - 1> operator[](uint32_t i)
        {
            const Shape<D> shape = static_cast<DERIVED*>(this)->getShape();
            DTYPE* ptr = static_cast<DERIVED*>(this)->getData();
            ptr += shape.getStride(0) * i;
            Shape<D - 1> out_shape = stripOuterDim(shape);
            return Tensor<DTYPE, D - 1>(ptr, std::move(out_shape));
        }
        template <class... ARGS>
        DTYPE& operator()(ARGS&&... args)
        {
            return *static_cast<DERIVED*>(this)->ptr(std::forward<ARGS>(args)...);
        }
    };

    template <class DERIVED, class DTYPE>
    class TensorIndexing<DERIVED, DTYPE, 1, typename std::enable_if<!std::is_const<DTYPE>::value>::type>
        : public ConstTensorIndexing<DERIVED, DTYPE, 1>
    {
      public:
        DTYPE& operator[](uint32_t i) { return *static_cast<DERIVED*>(this)->ptr(i); }
        template <class... ARGS>
        DTYPE& operator()(ARGS&&... args)
        {
            return *static_cast<DERIVED*>(this)->ptr(std::forward<ARGS>(args)...);
        }
    };

    template <class DERIVED, uint8_t D>
    class TensorIndexing<DERIVED, void, D>
    {
    };

    template <class DERIVED, uint8_t D>
    class TensorIndexing<DERIVED, const void, D>
    {
    };

    template <class DERIVED, class DTYPE, uint8_t D>
    class TensorIndexing<DERIVED, const DTYPE, D> : public ConstTensorIndexing<DERIVED, DTYPE, D>
    {
    };

    template <class T, uint8_t D>
    class Tensor<T, D, typename std::enable_if<!std::is_same<typename std::remove_const<T>::type, void>::value>::type>
        : public TensorIndexing<Tensor<T, D>, T, D>
    {
        T* m_ptr;
        Shape<D> m_shape;

      public:
        Tensor(T* ptr = nullptr, Shape<D> shape = Shape<D>()) : m_ptr(ptr), m_shape(shape) {}
        Tensor(Tensor<T, D>& other) : m_ptr(other.getData()), m_shape(other.getShape()) {}
        Tensor(const Tensor& other) = default;
        Tensor(Tensor&& other) = default;

        template <class... ARGS>
        T* ptr(ARGS&&... args)
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return &m_ptr[index];
        }

        template <class... ARGS>
        const T* ptr(ARGS&&... args) const
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return &m_ptr[index];
        }

        MT_XINLINE Shape<D> getShape() const { return m_shape; }
        MT_XINLINE const T* getData() const { return m_ptr; }
        MT_XINLINE T* getData() { return m_ptr; }
    };
    template <class T, uint8_t D>
    class Tensor<T, D, typename std::enable_if<std::is_same<typename std::remove_const<T>::type, void>::value>::type>
        : public TensorIndexing<Tensor<T, D>, T, D>
    {
        T* m_ptr;
        Shape<D> m_shape;

      public:
        Tensor(T* ptr = nullptr, Shape<D> shape = Shape<D>()) : m_ptr(ptr), m_shape(shape) {}

        template <class U>
        Tensor(Tensor<U, D>& other) : m_ptr(static_cast<void*>(other.getData()))
        {
            const Shape<D>& other_shape = other.getShape();
            m_shape = copyScaled<sizeof(U), 1>(other_shape);
        }

        template <class U>
        Tensor(Tensor<U, D>&& other) : m_ptr(static_cast<void*>(other.getData()))
        {
            const Shape<D>& other_shape = other.getShape();
            m_shape = copyScaled<sizeof(U), 1>(other_shape);
        }

        Tensor(Tensor& other) = default;

        Tensor(Tensor&& other) = default;

        template <class... ARGS>
        const T* ptr(ARGS&&... args) const
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return m_ptr + index;
        }

        template <class... ARGS>
        T* ptr(ARGS&&... args)
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return m_ptr + index;
        }

        MT_XINLINE const Shape<D>& getShape() const { return m_shape; }
        MT_XINLINE const void* getData() const { return m_ptr; }
        MT_XINLINE void* getData() { return m_ptr; }

        template <class U>
        operator Tensor<U, D>()
        {
            Shape<D> out_shape = copyScaled<1, sizeof(U)>(m_shape);
            return Tensor<U, D>(static_cast<U*>(m_ptr), std::move(out_shape));
        }
    };

    template <class T, uint8_t D>
    class TensorIterator
    {
        T* m_ptr;
        ssize_t m_stride;
        Shape<D> m_shape;

      public:
        TensorIterator(T* ptr, ssize_t stride, Shape<D> shape) : m_ptr(ptr), m_stride(stride), m_shape(shape) {}

        Tensor<T, D> operator*() const { return Tensor<T, D>(m_ptr, m_shape); }

        TensorIterator& operator++() { m_ptr += m_stride; }
        bool operator!=(const TensorIterator& other) const { return m_ptr != other.m_ptr; }
    };

    template <class T>
    class TensorIterator<T, 0>
    {
        T* m_ptr;
        size_t m_stride;

      public:
        TensorIterator(T* ptr, ssize_t stride, Shape<0>) : m_ptr(ptr), m_stride(stride) {}

        T& operator*() const { return *m_ptr; }

        TensorIterator& operator++() { m_ptr += m_stride; }

        bool operator!=(const TensorIterator& other) const { return m_ptr != other.m_ptr; }
    };

    template <class T, uint8_t D>
    TensorIterator<const T, D - 1> begin(const Tensor<T, D>& tensor)
    {
        const auto& shape = tensor.getShape();
        Shape<D - 1> out_shape = stripOuterDim(shape);
        const auto outer_stride = shape.getStride(0);
        auto ptr = tensor.getData();
        return TensorIterator<const T, D - 1>(ptr, outer_stride, out_shape);
    }

    template <class T, uint8_t D>
    TensorIterator<T, D - 1> begin(Tensor<T, D>& tensor)
    {
        const auto& shape = tensor.getShape();
        Shape<D - 1> out_shape = stripOuterDim(shape);
        const auto outer_stride = shape.getStride(0);
        auto ptr = tensor.getData();
        return TensorIterator<T, D - 1>(ptr, outer_stride, out_shape);
    }

    template <class T, uint8_t D>
    TensorIterator<const T, D - 1> end(const Tensor<T, D>& tensor)
    {
        const auto shape = tensor.getShape();
        Shape<D - 1> out_shape = stripOuterDim(shape);
        const auto outer_stride = shape.getStride(0);
        const auto step = outer_stride * shape[0];
        auto ptr = tensor.getData();
        ptr += step;
        return TensorIterator<const T, D - 1>(ptr, outer_stride, out_shape);
    }

    template <class T, uint8_t D>
    TensorIterator<T, D - 1> end(Tensor<T, D>& tensor)
    {
        const auto shape = tensor.getShape();
        Shape<D - 1> out_shape = stripOuterDim(shape);
        const auto outer_stride = shape.getStride(0);
        const auto step = outer_stride * shape[0];
        auto ptr = tensor.getData();
        return TensorIterator<T, D - 1>(ptr + step, outer_stride, out_shape);
    }

    template <class T>
    void printTensor(std::ostream& os, const T& value, const std::string& = "  ")
    {
        os << value;
    }

    template <class T, uint8_t D>
    void printTensor(std::ostream& os, const mt::Tensor<T, D>& tensor, const std::string& indent = " ")
    {
        size_t i = 0;
        for (const auto& itr : tensor)
        {
            os << indent;
            printTensor(os, itr, indent + " ");
            ++i;
        }
        if (D >= 1)
        {
            os << "\n";
        }
    }

} // namespace mt

namespace std
{

    template <class T, uint8_t D>
    ostream& operator<<(ostream& os, const mt::Tensor<T, D>& tensor)
    {
        os << tensor.getShape();
        os << "\nDataType: " << typeid(T).name() << '\n';
        mt::printTensor(os, tensor, std::string());
        return os;
    }
} // namespace std

#endif // MINITENSOR_TENSOR_HPP