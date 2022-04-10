#ifndef MINITENSOR_TENSOR_HPP
#define MINITENSOR_TENSOR_HPP
#include "defines.hpp"

#include "Shape.hpp"
#include "utilities.hpp"

#include <ct/reflect_traits.hpp>
#include <ct/types/TArrayView.hpp>

#include <assert.h>
#include <cstddef>
#include <typeinfo>
#include <vector>

namespace mt
{
    template <class T>
    struct SafeSizeOf
    {
        static constexpr const size_t value = sizeof(T);
    };

    template <>
    struct SafeSizeOf<void>
    {
        // 1 byte
        static constexpr const size_t value = 1;
    };

    template <>
    struct SafeSizeOf<const void>
    {
        // 1 byte
        static constexpr const size_t value = 1;
    };

    template <class T, uint8_t D, template <uint8_t> class LAYOUT = Shape>
    class Tensor;

    template <class T, class U>
    struct IsSame;

    // Helper function for checking if all types in a typedef are the same as the query T
    template <class T, class... Us>
    struct IsSame<T, ct::VariadicTypedef<Us...>>
    {
        template <class U>
        using Checker = std::is_same<T, U>;
        static constexpr const bool value = ct::VariadicTypedef<Us...>::template all<Checker>();
    };

    ////////////////////////////////////////////////////////////////////////////
    ///  Tensor class
    ////////////////////////////////////////////////////////////////////////////

    /**
     * @brief The Tensor class
     */
    template <class T, uint8_t D, template <uint8_t> class LAYOUT>
    class Tensor
    {
        T* m_ptr;
        LAYOUT<D> m_shape;

        /*template<class U>
        using IsSame = std::is_same<T, U>;*/

      public:
        static constexpr const uint8_t DIM = D;
        using DType = T;

        Tensor(T* ptr = nullptr, Shape<D> shape = Shape<D>()) : m_ptr(ptr), m_shape(shape) {}

        Tensor(Tensor<T, D>& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(Tensor&& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        template <uint8_t D1>
        Tensor(Tensor<DType, D1, LAYOUT>& other)
        {
            static_assert(
                D > D1,
                "Can only construct a larger tensor from a smaller one by implicitly unsqueezing outer dimensions");
            m_shape = other.getShape();
            m_ptr = other.data();
        }

        template <class DType>
        Tensor(Tensor<DType, D + 1, LAYOUT>& raw_data_tensor,
               typename std::enable_if<!std::is_same<DType, T>::value>::type* = nullptr)
        {
            using types = typename ct::GlobMemberObjects<T>::types;
            static_assert(IsSame<DType, types>::value, "");
            static_assert(sizeof(T) == sizeof(DType) * types::size(),
                          "All parts of the object must be considered for reflection");
            const auto& shape = raw_data_tensor.getShape();
            if (shape.getStride(D) != 1)
            {
                throw std::runtime_error(
                    "Trying to recover a tensor of an aggregate type from a raw tensor with non continuous stride");
            }
            m_ptr = ct::ptrCast<T>(raw_data_tensor.data());
            for (uint32_t i = 0; i < D; ++i)
            {
                m_shape.setShape(i, shape[i]);
                m_shape.setStride(i, shape.getStride(i) / types::size());
            }
        }

        template <class OBJ>
        Tensor(Tensor<OBJ, D - 1, LAYOUT>& aggregate_type_tensor,
               typename std::enable_if<!std::is_same<OBJ, T>::value>::type* = nullptr)
        {
            using types = typename ct::GlobMemberObjects<OBJ>::types;
            static_assert(IsSame<T, types>::value,
                          "All types of the aggregate object must match type of wrapping tensor");
            static_assert(sizeof(OBJ) == sizeof(T) * types::size(),
                          "All parts of the object must be considered for reflection");
            m_ptr = ct::ptrCast<T>(aggregate_type_tensor.data());
            const auto& shape = aggregate_type_tensor.getShape();
            for (uint8_t i = 0; i < D - 1; ++i)
            {
                m_shape.setShape(i, shape[i]);
                m_shape.setStride(i, shape.getStride(i) * types::size());
            }
            m_shape.setShape(D - 1, types::size());
            m_shape.setStride(D - 1, 1);
        }

        Tensor& operator=(const Tensor& other)
        {
            // Copy contents from other into this
            const size_t outer_dim = m_shape[0];
            assert(outer_dim == other.m_shape[0]);
            for (size_t i = 0; i < outer_dim; ++i)
            {
                (*this)[i] = other[i];
            }
            return *this;
        }

        Tensor& operator=(Tensor&&) = default;

        Tensor<T, D - 1, LAYOUT> slice(int32_t dim, int32_t index)
        {
            dim = revIndex(dim, D);
            index = revIndex(index, m_shape[dim]);
            Shape<D - 1> shape;
            const size_t offset = m_shape.getStride(dim) * index;
            for (uint8_t i = 0; i < dim; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            for (uint8_t i = dim; i < (D - 1); ++i)
            {
                shape.setShape(i, m_shape[i + 1]);
                shape.setStride(i, m_shape.getStride(i + 1));
            }
            return Tensor<T, D - 1, LAYOUT>(m_ptr + offset, shape);
        }

        Tensor<const T, D - 1, LAYOUT> slice(int32_t dim, int32_t index) const
        {
            dim = revIndex(dim, D);
            index = revIndex(index, m_shape[dim]);
            Shape<D - 1> shape;
            const size_t offset = m_shape.getStride(dim) * index;
            for (uint8_t i = 0; i < dim; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            for (uint8_t i = dim; i < (D - 1); ++i)
            {
                shape.setShape(i, m_shape[i + 1]);
                shape.setStride(i, m_shape.getStride(i + 1));
            }
            return Tensor<const T, D - 1, LAYOUT>(m_ptr + offset, shape);
        }

        Tensor<T, D, LAYOUT> slice(int32_t dim, int32_t begin, int32_t end)
        {
            dim = revIndex(dim, D);
            begin = revIndex(begin, m_shape[dim]);
            end = revIndex(end, m_shape[dim]);
            Shape<D - 1> shape;
            const size_t offset = m_shape.getStride(dim) * begin;
            for (uint8_t i = 0; i < dim; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            shape.setShape(dim, end - begin);
            shape.setStride(dim, m_shape.getStride(dim));
            for (uint8_t i = dim; i < D; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            return Tensor<T, D, LAYOUT>(m_ptr + offset, shape);
        }

        Tensor<const T, D, LAYOUT> slice(int32_t dim, int32_t begin, int32_t end) const
        {
            dim = revIndex(dim, D);
            begin = revIndex(begin, m_shape[dim]);
            end = revIndex(end, m_shape[dim]);
            Shape<D - 1> shape;
            const size_t offset = m_shape.getStride(dim) * begin;
            for (uint8_t i = 0; i < dim; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            shape.setShape(dim, end - begin);
            shape.setStride(dim, m_shape.getStride(dim));
            for (uint8_t i = dim; i < D; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            return Tensor<const T, D, LAYOUT>(m_ptr + offset, shape);
        }

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

        template <class... ARGS>
        T& operator()(ARGS&&... args)
        {
            return *ptr(std::forward<ARGS>(args)...);
        }

        template <class... ARGS>
        const T& operator()(ARGS&&... args) const
        {
            return *ptr(std::forward<ARGS>(args)...);
        }

        MT_XINLINE const LAYOUT<D>& getShape() const { return m_shape; }

        MT_XINLINE const T* data() const { return m_ptr; }

        MT_XINLINE T* data() { return m_ptr; }

        Tensor<const T, D - 1> operator[](const size_t idx) const
        {
            Shape<D - 1> out_shape = m_shape.minorShape();
            const size_t offset = m_shape.index(idx);
            const T* ptr = m_ptr + offset;
            return Tensor<const T, D - 1>(ptr, out_shape);
        }

        Tensor<T, D - 1> operator[](const size_t idx)
        {
            Shape<D - 1> out_shape = m_shape.minorShape();
            const size_t offset = m_shape.index(idx);
            T* ptr = m_ptr + offset;
            return Tensor<T, D - 1>(ptr, out_shape);
        }
    };

    template <class T, uint8_t D, template <uint8_t> class LAYOUT>
    class Tensor<const T, D, LAYOUT>
    {
        const T* m_ptr;
        LAYOUT<D> m_shape;
        template <class U>
        using IsSame = std::is_same<T, U>;

      public:
        static constexpr const uint8_t DIM = D;
        using DType = T;

        Tensor(const T* ptr = nullptr, Shape<D> shape = Shape<D>()) : m_ptr(ptr), m_shape(shape) {}

        Tensor(const Tensor<const T, D, LAYOUT>& other) = default;

        Tensor(const Tensor<T, D>& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(Tensor&& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        template <uint8_t D1>
        Tensor(const Tensor<const DType, D1, LAYOUT>& other)
        {
            static_assert(
                D > D1,
                "Can only construct a larger tensor from a smaller one by implicitly unsqueezing outer dimensions");
            m_shape = other.getShape();
            m_ptr = other.data();
        }

        template <class OBJ>
        Tensor(Tensor<OBJ, D - 1, LAYOUT>& aggregate_type_tensor)
        {
            using types = typename ct::GlobMemberObjects<OBJ>::types;
            static_assert(types::template all<IsSame>(),
                          "All types of the aggregate object must match type of wrapping tensor");
            static_assert(sizeof(OBJ) == sizeof(T) * types::size(),
                          "All parts of the object must be considered for reflection");
            m_ptr = ct::ptrCast<T>(aggregate_type_tensor.data());
            const auto& shape = aggregate_type_tensor.getShape();
            for (uint8_t i = 0; i < D - 1; ++i)
            {
                m_shape.setShape(i, shape[i]);
                m_shape.setStride(i, shape.getStride(i) * types::size());
            }
            m_shape.setShape(D - 1, types::size());
            m_shape.setStride(D - 1, 1);
        }

        Tensor& operator=(const Tensor& other)
        {
            // Copy contents from other into this
            const size_t outer_dim = m_shape[0];
            assert(outer_dim == other.m_shape[0]);
            for (size_t i = 0; i < outer_dim; ++i)
            {
                (*this)[i] = other[i];
            }
            return *this;
        }

        Tensor& operator=(Tensor&&) = default;

        template <class... ARGS>
        const T* ptr(ARGS&&... args) const
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return &m_ptr[index];
        }

        template <class... ARGS>
        const T& operator()(ARGS&&... args) const
        {
            return *ptr(std::forward<ARGS>(args)...);
        }

        Tensor<const T, D - 1, LAYOUT> slice(int32_t dim, int32_t index) const
        {
            dim = revIndex(dim, D);
            index = revIndex(index, m_shape[dim]);
            Shape<D - 1> shape;
            const size_t offset = m_shape.getStride(dim) * index;
            for (uint8_t i = 0; i < dim; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            for (uint8_t i = dim; i < (D - 1); ++i)
            {
                shape.setShape(i, m_shape[i + 1]);
                shape.setStride(i, m_shape.getStride(i + 1));
            }
            return Tensor<const T, D - 1, LAYOUT>(m_ptr + offset, shape);
        }

        Tensor<const T, D, LAYOUT> slice(int32_t dim, int32_t begin, int32_t end) const
        {
            dim = revIndex(dim, D);
            begin = revIndex(begin, m_shape[dim]);
            end = revIndex(end, m_shape[dim]);
            Shape<D - 1> shape;
            const size_t offset = m_shape.getStride(dim) * begin;
            for (uint8_t i = 0; i < dim; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            shape.setShape(dim, end - begin);
            shape.setStride(dim, m_shape.getStride(dim));
            for (uint8_t i = dim; i < D; ++i)
            {
                shape.setShape(i, m_shape[i]);
                shape.setStride(i, m_shape.getStride(i));
            }
            return Tensor<const T, D, LAYOUT>(m_ptr + offset, shape);
        }

        MT_XINLINE const LAYOUT<D>& getShape() const { return m_shape; }

        MT_XINLINE const T* data() const { return m_ptr; }

        Tensor<const T, D - 1> operator[](const size_t idx) const
        {
            LAYOUT<D - 1> out_shape = m_shape.minorShape();
            const size_t offset = m_shape.index(idx);
            const T* ptr = m_ptr + offset;
            return Tensor<const T, D - 1>(ptr, out_shape);
        }
    };

    /**
     * @brief The Tensor class 1d mutable specialization
     */
    template <class T, template <uint8_t> class LAYOUT>
    class Tensor<T, 1, LAYOUT>
    {
        T* m_ptr;
        LAYOUT<1> m_shape;

      public:
        static constexpr const uint8_t DIM = 1;
        using DType = T;

        Tensor(T* ptr = nullptr, Shape<1> shape = Shape<1>()) : m_ptr(ptr), m_shape(shape) {}

        Tensor(Tensor<T, 1>& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(const Tensor& other) : m_ptr(other.m_ptr), m_shape(other.getShape()) {}

        Tensor(Tensor&& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(std::vector<T>& vec) : m_ptr(vec.data()), m_shape(vec.size()) {}

        Tensor(ct::TArrayView<T> view) : m_ptr(view.data()), m_shape(view.size()) {}

        Tensor& operator=(Tensor&&) = default;

        Tensor& operator=(const std::vector<T>& data)
        {
            if (data.size() != m_shape.numElements())
            {
                throw std::runtime_error("Input data vector does not match size of tensor, cannot copy elements");
            }
            const size_t size = m_shape.numElements();
            for (size_t i = 0; i < size; ++i)
            {
                // This accounts for reverse indexing, etc
                const size_t idx = m_shape.index(i);
                m_ptr[idx] = data[i];
            }
            return *this;
        }

        T& slice(int32_t, int32_t index)
        {
            index = revIndex(index, m_shape[0]);
            (*this)[index];
        }

        const T& slice(int32_t, int32_t index) const
        {
            index = revIndex(index, m_shape[0]);
            (*this)[index];
        }

        Tensor<T, 1, LAYOUT> slice(int32_t, int32_t begin, int32_t end)
        {
            begin = revIndex(begin, m_shape[0]);
            end = revIndex(end, m_shape[0]);
            Shape<1> shape = m_shape;
            shape.setShape(0, end - begin + 1);
            return Tensor<T, 1, LAYOUT>(ptr(begin), shape);
        }

        Tensor<const T, 1, LAYOUT> slice(int32_t, int32_t begin, int32_t end) const
        {
            begin = revIndex(begin, m_shape[0]);
            end = revIndex(end, m_shape[0]);
            Shape<1> shape = m_shape;
            shape.setShape(0, end - begin + 1);
            return Tensor<const T, 1, LAYOUT>(ptr(begin), shape);
        }

        Tensor& operator=(const Tensor& other)
        {
            const uint32_t N = std::min(m_shape[0], other.m_shape[0]);
            for (uint32_t i = 0; i < N; ++i)
            {
                (*this)[i] = other[i];
            }
            return *this;
        }

        Tensor& operator=(const Tensor<const T, 1>& other)
        {
            const uint32_t N = std::min(m_shape[0], other.getShape()[0]);
            for (uint32_t i = 0; i < N; ++i)
            {
                (*this)[i] = other[i];
            }
            return *this;
        }

        T* ptr(const uint32_t idx)
        {
            const size_t index = m_shape.index(idx);
            return &m_ptr[index];
        }

        const T* ptr(const uint32_t idx) const
        {
            const size_t index = m_shape.index(idx);
            return &m_ptr[index];
        }

        T& operator()(const uint32_t idx) { return *ptr(idx); }

        const T& operator()(const uint32_t idx) const { return *ptr(idx); }

        MT_XINLINE const LAYOUT<1>& getShape() const { return m_shape; }

        MT_XINLINE const T* data() const { return m_ptr; }

        MT_XINLINE T* data() { return m_ptr; }

        const T& operator[](const size_t idx) const { return *ptr(idx); }

        T& operator[](const size_t idx) { return *ptr(idx); }

        operator ct::TArrayView<const T>() const
        {
            // TODO assert on continuous
            return ct::TArrayView<const T>(m_ptr, m_shape[0]);
        }

        operator ct::TArrayView<T>()
        {
            // TODO assert on continuous
            return ct::TArrayView<T>(m_ptr, m_shape[0]);
        }
    };

    template <class T, template <uint8_t> class LAYOUT>
    class Tensor<const T, 1, LAYOUT>
    {
        const T* m_ptr;
        LAYOUT<1> m_shape;

      public:
        static constexpr const uint8_t DIM = 1;
        using DType = T;

        Tensor(const T* ptr = nullptr, Shape<1> shape = Shape<1>()) : m_ptr(ptr), m_shape(shape) {}

        /**
         * @brief Tensor copy from const tensor to non const data
         * @param other
         */
        Tensor(const Tensor<T, 1>& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(const Tensor& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(Tensor&& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(Tensor<T, 1>&& other) : m_ptr(other.data()), m_shape(other.getShape()) {}

        Tensor(const std::vector<T>& vec) : m_ptr(vec.data()), m_shape(vec.size()) {}

        Tensor(const ct::TArrayView<const T> view) : m_ptr(view.data()), m_shape(view.size()) {}

        Tensor(const ct::TArrayView<T> view) : m_ptr(view.data()), m_shape(view.size()) {}

        Tensor& operator=(Tensor&&) = default;

        const T* ptr(const uint32_t idx) const
        {
            const size_t index = m_shape.index(idx);
            return &m_ptr[index];
        }

        const T& operator()(const uint32_t idx) const { return *ptr(idx); }

        const T& slice(int32_t, int32_t index) const
        {
            index = revIndex(index, m_shape[0]);
            (*this)[index];
        }

        Tensor<const T, 1, LAYOUT> slice(int32_t, int32_t begin, int32_t end) const
        {
            begin = revIndex(begin, m_shape[0]);
            end = revIndex(end, m_shape[0]);
            Shape<1> shape = m_shape;
            shape.setShape(0, end - begin + 1);
            return Tensor<const T, 1, LAYOUT>(ptr(begin), shape);
        }

        MT_XINLINE const Shape<1>& getShape() const { return m_shape; }

        MT_XINLINE const T* data() const { return m_ptr; }

        const T& operator[](const uint32_t idx) const { return *ptr(idx); }

        operator ct::TArrayView<const T>() const
        {
            // TODO assert on continuous
            return ct::TArrayView<const T>(m_ptr, m_shape[0]);
        }
    };

    /**
     * @brief The Tensor class void specialization
     */
    template <uint8_t D, template <uint8_t> class LAYOUT>
    class Tensor<void, D, LAYOUT>
    {
        void* m_ptr;
        LAYOUT<D> m_shape;

      public:
        static constexpr const uint8_t DIM = D;
        using DType = void;

        Tensor(void* ptr = nullptr, Shape<D> shape = Shape<D>()) : m_ptr(ptr), m_shape(shape) {}

        template <class U>
        Tensor(Tensor<U, D, LAYOUT>& other) : m_ptr(static_cast<void*>(other.data()))
        {
            const LAYOUT<D>& other_shape = other.getShape();
            m_shape = copyScaled<sizeof(U), 1>(other_shape);
        }

        template <class U>
        Tensor(Tensor<U, D, LAYOUT>&& other) : m_ptr(static_cast<void*>(other.data()))
        {
            const LAYOUT<D>& other_shape = other.getShape();
            m_shape = copyScaled<sizeof(U), 1>(other_shape);
        }

        template <class U, uint8_t D2>
        Tensor(Tensor<U, D2, LAYOUT>&& other) : m_ptr(static_cast<void*>(other.data()))
        {
            const LAYOUT<D2>& other_shape = other.getShape();
            m_shape = copyScaled<sizeof(U), 1>(other_shape);
        }

        Tensor(Tensor& other) = default;

        Tensor(Tensor&& other) = default;

        Tensor& operator=(const Tensor& other)
        {
            // Copy contents from other into this
            const size_t outer_dim = m_shape[0];
            assert(outer_dim == other.m_shape[0]);
            for (size_t i = 0; i < outer_dim; ++i)
            {
                (*this)[i] = other[i];
            }
            return *this;
        }

        Tensor& operator=(Tensor&&) = default;

        template <class... ARGS>
        const void* ptr(ARGS&&... args) const
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return ct::ptrCast<void>(ct::ptrCast<uint8_t>(m_ptr) + index);
        }

        template <class... ARGS>
        void* ptr(ARGS&&... args)
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return ct::ptrCast<void>(ct::ptrCast<uint8_t>(m_ptr) + index);
        }

        MT_XINLINE const LAYOUT<D>& getShape() const { return m_shape; }

        MT_XINLINE const void* data() const { return m_ptr; }

        MT_XINLINE void* data() { return m_ptr; }

        template <class U>
        operator Tensor<U, D>()
        {
            LAYOUT<D> out_shape = copyScaled<1, sizeof(U)>(m_shape);
            return Tensor<U, D>(static_cast<U*>(m_ptr), std::move(out_shape));
        }

        template <class U>
        operator Tensor<const U, D>() const
        {
            LAYOUT<D> out_shape = copyScaled<1, sizeof(U)>(m_shape);
            return Tensor<U, D>(static_cast<const U*>(m_ptr), std::move(out_shape));
        }
    };

    /**
     * @brief The Tensor class const void specialization
     */
    template <uint8_t D, template <uint8_t> class LAYOUT>
    class Tensor<const void, D, LAYOUT>
    {
        const void* m_ptr;
        LAYOUT<D> m_shape;

      public:
        static constexpr const uint8_t DIM = D;
        using DType = void;

        Tensor(const void* ptr = nullptr, LAYOUT<D> shape = LAYOUT<D>()) : m_ptr(ptr), m_shape(shape) {}

        Tensor(Tensor<const void, D, LAYOUT>& other) = default;

        Tensor(Tensor<const void, D, LAYOUT>&& other) = default;

        template <class U, uint8_t D2, template <uint8_t> class L2>
        Tensor(Tensor<U, D2, L2>& other) : m_ptr(static_cast<const void*>(other.data()))
        {
            const LAYOUT<D>& other_shape = other.getShape();
            m_shape = copyScaled<SafeSizeOf<U>::value, 1>(other_shape);
        }

        template <class U, uint8_t D2, template <uint8_t> class L2>
        Tensor(Tensor<U, D2, L2>&& other) : m_ptr(static_cast<const void*>(other.data()))
        {
            const LAYOUT<D>& other_shape = other.getShape();
            m_shape = copyScaled<SafeSizeOf<U>::value, 1>(other_shape);
        }

        Tensor& operator=(const Tensor&) = delete;

        Tensor& operator=(Tensor&&) = default;

        template <class... ARGS>
        const void* ptr(ARGS&&... args) const
        {
            const size_t index = m_shape.index(std::forward<ARGS>(args)...);
            return ct::ptrCast<void>(ct::ptrCast<uint8_t>(m_ptr) + index);
        }

        MT_XINLINE const LAYOUT<D>& getShape() const { return m_shape; }

        MT_XINLINE const void* data() const { return m_ptr; }

        template <class U>
        operator Tensor<const U, D, LAYOUT>() const
        {
            LAYOUT<D> out_shape = copyScaled<1, sizeof(U)>(m_shape);
            return Tensor<const U, D>(static_cast<const U*>(m_ptr), std::move(out_shape));
        }
    };

    // scalar value specialization
    template <class T, template <uint8_t> class LAYOUT>
    class Tensor<T, 0, LAYOUT>
    {
        T* m_ptr;

      public:
        static constexpr const uint8_t DIM = 0;
        using DType = T;

        Tensor(T* ptr = nullptr, LAYOUT<0> = LAYOUT<0>()) : m_ptr(ptr) {}

        template <class... ARGS>
        const T* ptr(ARGS&&...) const
        {
            return m_ptr;
        }

        template <class... ARGS>
        T* ptr(ARGS&&...)
        {
            return m_ptr;
        }

        T& operator[](size_t) { return *m_ptr; }

        const T& operator[](size_t) const { return *m_ptr; }

        void copyTo(Tensor<typename std::remove_const<T>::type, 0> dst) const { *dst.data() = *m_ptr; }

        template <class U>
        void copyTo(U&& dst) const
        {
            dst = *m_ptr;
        }

        MT_XINLINE LAYOUT<0> getShape() const { return LAYOUT<0>(); }

        MT_XINLINE const T* data() const { return m_ptr; }

        MT_XINLINE T* data() { return m_ptr; }

        template <class U>
        operator U&()
        {
            return *m_ptr;
        }

        template <class U>
        operator U() const
        {
            return *m_ptr;
        }
    };

    template <class T, template <uint8_t> class LAYOUT>
    class Tensor<const T, 0, LAYOUT>
    {
        const T* m_ptr;

      public:
        static constexpr const uint8_t DIM = 0;
        using DType = T;

        Tensor(const T* ptr = nullptr, LAYOUT<0> = LAYOUT<0>()) : m_ptr(ptr) {}

        template <class... ARGS>
        const T* ptr(ARGS&&...) const
        {
            return m_ptr;
        }

        const T& operator[](size_t) const { return *m_ptr; }

        void copyTo(Tensor<typename std::remove_const<T>::type, 0> dst) const { *dst.data() = *m_ptr; }

        template <class U>
        void copyTo(U&& dst) const
        {
            dst = *m_ptr;
        }

        MT_XINLINE LAYOUT<0> getShape() const { return LAYOUT<0>(); }

        MT_XINLINE const T* data() const { return m_ptr; }

        template <class U>
        operator U() const
        {
            return *m_ptr;
        }

        /*operator T() const
        {
            return *m_ptr;
        }*/
    };

    ///////////////////////////////////////////////////////////////////////////
    //            TensorIterator
    ///////////////////////////////////////////////////////////////////////////
    template <class T, uint8_t D>
    class TensorIterator
    {
        T* m_ptr;
        int64_t m_stride;
        Shape<D> m_shape;

      public:
        TensorIterator(T* ptr, int64_t stride, Shape<D> shape) : m_ptr(ptr), m_stride(stride), m_shape(shape) {}

        Tensor<T, D> operator*() const { return Tensor<T, D>(m_ptr, m_shape); }

        TensorIterator& operator++()
        {
            m_ptr += m_stride;
            return *this;
        }
        bool operator!=(const TensorIterator& other) const { return m_ptr != other.m_ptr; }
    };

    template <class T>
    class TensorIterator<T, 0>
    {
        T* m_ptr;
        size_t m_stride;

      public:
        TensorIterator(T* ptr, int64_t stride, Shape<0>) : m_ptr(ptr), m_stride(stride) {}

        T& operator*() const { return *m_ptr; }

        TensorIterator& operator++()
        {
            m_ptr += m_stride;
            return *this;
        }

        bool operator!=(const TensorIterator& other) const { return m_ptr != other.m_ptr; }
    };

    template <class T, uint8_t D>
    TensorIterator<const T, D - 1> begin(const Tensor<T, D>& tensor)
    {
        const auto& shape = tensor.getShape();
        Shape<D - 1> out_shape = stripOuterDim(shape);
        const auto outer_stride = shape.getStride(0);
        auto ptr = tensor.data();
        return TensorIterator<const T, D - 1>(ptr, outer_stride, out_shape);
    }

    template <class T, uint8_t D>
    TensorIterator<T, D - 1> begin(Tensor<T, D>& tensor)
    {
        const auto& shape = tensor.getShape();
        Shape<D - 1> out_shape = stripOuterDim(shape);
        const auto outer_stride = shape.getStride(0);
        auto ptr = tensor.data();
        return TensorIterator<T, D - 1>(ptr, outer_stride, out_shape);
    }

    template <class T, uint8_t D>
    TensorIterator<const T, D - 1> end(const Tensor<T, D>& tensor)
    {
        const auto shape = tensor.getShape();
        Shape<D - 1> out_shape = stripOuterDim(shape);
        const auto outer_stride = shape.getStride(0);
        const auto step = outer_stride * shape[0];
        auto ptr = tensor.data();
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
        auto ptr = tensor.data();
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

    template <class T, class E = void, uint8_t P = 5>
    struct TensorWrap : TensorWrap<T, E, P - 1>
    {
    };

    template <class T>
    struct TensorWrap<T, void, 0>
    {
        static Tensor<T, 0> wrap(T& data) { return Tensor<T, 0>(&data); }
        static Tensor<const T, 0> wrap(const T& data) { return Tensor<const T, 0>(&data); }
    };

    template <class T>
    struct TensorWrap<T, typename std::enable_if<std::is_const<T>::value>::type, 1>
    {
        static Tensor<T, 0> wrap(T& data) { return Tensor<T, 0>(&data); }
    };

    template <class T>
    struct TensorWrap<std::vector<T>, void, 2>
    {

        static Tensor<const T, 1> wrap(const std::vector<T>& data)
        {
            return Tensor<const T, 1>(data.data(), data.size());
        }

        static Tensor<T, 1> wrap(std::vector<T>& data) { return Tensor<T, 1>(data.data(), data.size()); }
    };

    template <class T>
    auto tensorWrap(const T& data) -> decltype(TensorWrap<T>::wrap(data))
    {
        return TensorWrap<T>::wrap(data);
    }

    template <class T>
    auto tensorWrap(T& data) -> decltype(TensorWrap<T>::wrap(data))
    {
        return TensorWrap<T>::wrap(data);
    }

    template <class T, uint8_t N, template <uint8_t> class LAYOUT>
    auto tensorWrap(const Tensor<T, N, LAYOUT>& data) -> Tensor<T, N, LAYOUT>
    {
        return data;
    }

    template <class T, uint8_t N, template <uint8_t> class LAYOUT>
    auto tensorWrap(Tensor<T, N, LAYOUT>& data) -> Tensor<T, N, LAYOUT>
    {
        return data;
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
