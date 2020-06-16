#ifndef MINITENSOR_SHAPE_HPP
#define MINITENSOR_SHAPE_HPP
#include "Array.hpp"

namespace mt
{
    template <uint8_t N>
    class Shape
    {
        Array<uint32_t, N> m_size;
        Array<int32_t, N> m_stride;

        void calculateStride()
        {
            // Assume last dim is densly packed
            m_stride[N - 1] = 1;
            for (int32_t i = N - 2; i >= 0; --i)
            {
                const auto prev_stride = m_stride[i + 1];
                const auto prev_shape = m_size[i + 1];
                m_stride[i] = prev_stride * prev_shape;
            }
        }

        template <uint8_t D, class T>
        size_t indexHelper(size_t out, T&& arg) const
        {
            out += m_stride[D] * revIndex(arg, m_size[D]);
            return out;
        }

        template <uint8_t D, class T, class... Ts>
        size_t indexHelper(size_t out, T&& arg, Ts&&... args) const
        {
            out += m_stride[D] * revIndex(arg, m_size[D]);
            return indexHelper<D + 1>(out, std::forward<Ts>(args)...);
        }

      public:
        Shape(const Shape& other) = default;
        Shape(Shape& other) = default;
        Shape(Shape&& other) = default;

        Shape& operator=(const Shape& other) = default;
        Shape& operator=(Shape&& other) = default;

        template <class... T>
        Shape(T&&... args) : m_size(std::forward<T>(args)...)
        {
            calculateStride();
        }

        template <class... T>
        size_t index(T&&... args) const
        {
            size_t out = 0;
            return indexHelper<0>(out, std::forward<T>(args)...);
        }

        MT_XINLINE uint32_t operator[](int16_t idx) const { return m_size[idx]; }
        MT_XINLINE uint32_t getStride(int16_t idx) const { return m_stride[idx]; }

        void setShape(uint8_t dim, uint32_t size) { m_size[dim] = size; }
        void setStride(uint8_t dim, uint32_t stride) { m_stride[dim] = stride; }
        bool operator==(const Shape& other) const
        {
            for (uint8_t i = 0; i < N; ++i)
            {
                if (m_size[i] != other[i])
                {
                    return false;
                }
            }
            return true;
        }
    };

    template <uint8_t N>
    Shape<N - 1> stripOuterDim(const Shape<N>& shape)
    {
        Shape<N - 1> out;
        for (uint8_t d = 1; d < N; ++d)
        {
            out.setShape(d - 1, shape[d]);
            out.setStride(d - 1, shape.getStride(d));
        }
        return out;
    }
} // namespace mt

#include <ostream>

namespace std
{
    template <uint8_t N>
    ostream& operator<<(ostream& os, const mt::Shape<N>& arr)
    {
        os << "size: ";
        for (uint8_t i = 0; i < N; ++i)
        {
            if (i != 0)
            {
                os << ' ';
            }
            os << arr[i];
        }
        os << "\nstride: ";
        for (uint8_t i = 0; i < N; ++i)
        {
            if (i != 0)
            {
                os << ' ';
            }
            os << arr.getStride(i);
        }
        return os;
    }
} // namespace std

#endif // MINITENSOR_SHAPE_HPP