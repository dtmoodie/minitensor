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
        Shape(const Shape& other):
            m_size(other.m_size), m_stride(other.m_stride)
        {

        }
        Shape(Shape& other):
            m_size(other.m_size), m_stride(other.m_stride)
        {

        }
        Shape(Shape&& other):
            m_size(other.m_size), m_stride(other.m_stride)
        {

        }

        Shape& operator=(const Shape& other) = default;
        Shape& operator=(Shape&& other) = default;

        template<uint8_t N1>
        Shape(Shape<N1>& other)
        {
            static_assert(N1 < N,"");
            const uint8_t padded = N - N1;
            int16_t i;
            for(i = 0; i < padded; ++i)
            {
                m_size[i] = 1;
            }
            for(; i < N; ++i)
            {
                m_size[i] = other[i - padded];
                m_stride[i] = other.getStride(i - padded);
            }
            for(i = padded - 1; i >= 0; --i)
            {
                m_stride[i] = m_size[i + 1] * m_stride[i + 1];
            }
        }

        template<uint8_t N1>
        Shape(const Shape<N1>& other)
        {
            static_assert(N1 < N,"");
            const uint8_t padded = N - N1;
            int16_t i;
            for(i = 0; i < padded; ++i)
            {
                m_size[i] = 1;
            }
            for(; i < N; ++i)
            {
                m_size[i] = other[i - padded];
                m_stride[i] = other.getStride(i - padded);
            }
            for(i = padded - 1; i >= 0; --i)
            {
                m_stride[i] = m_size[i + 1] * m_stride[i + 1];
            }
        }

        template<uint8_t N1>
        Shape(Shape<N1>&& other)
        {
            static_assert(N1 < N,"");
            const uint8_t padded = N - N1;
            int16_t i;
            for(i = 0; i < padded; ++i)
            {
                m_size[i] = 1;
            }
            for(; i < N; ++i)
            {
                m_size[i] = other[i - padded];
                m_stride[i] = other.getStride(i - padded);
            }
            for(i = padded - 1; i >= 0; --i)
            {
                m_stride[i] = m_size[i + 1] * m_stride[i + 1];
            }
        }

        template <class... T>
        Shape(T&&... args) : m_size(std::forward<T>(args)...)
        {
            calculateStride();
        }

        template <class... T>
        auto index(T&&... args) const -> typename std::enable_if<sizeof...(args) != 1, size_t>::type
        {
            size_t out = 0;
            return indexHelper<0>(out, std::forward<T>(args)...);
        }

        size_t index(size_t idx) const
        {
            size_t out = m_stride[0] * idx;
            return out;
        }

        /**
         * @brief linearIndex calculates the linear index of the idx'th element as if this was a c-style array
         *        IE for a 4x4 matrix
         *        | 0,  1,  2, 3
         *        | 4,  5,  6, 7
         *        | 8,  9, 10, 11
         *        |12, 13, 14, 15
         *        The 5th element has a value of 5 and 2d index of 1,1.  Linear index will return the offset from the beginning of the data
         *        block to where the element is.  This important distinction is due to the fact that reversed
         * @param idx
         * @return
         */
        size_t linearIndex(size_t idx) const
        {
            size_t out = 0;
            for (uint8_t i = 0; i < N - 1; ++i)
            {
                const size_t this_idx = idx / m_size[i + 1];
                idx = idx % m_size[i + 1];
                out += m_stride[i] * this_idx;
            }
            out += idx;
            return out;
        }

        Shape<N - 1> minorShape() const
        {
            Shape<N - 1> output;
            for(uint8_t i = 1; i < N; ++i)
            {
                output.setStride(i - 1, m_stride[i]);
                output.setShape(i - 1, m_size[i]);
            }
            return output;
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

        void calculateStride()
        {
            // Assume last dim is densly packed
            m_stride[N - 1] = 1;
            for (int16_t i = N - 2; i >= 0; --i)
            {
                const auto prev_stride = m_stride[i + 1];
                const auto prev_shape = m_size[i + 1];
                m_stride[i] = prev_stride * prev_shape;
            }
        }

        size_t numElements() const
        {
            size_t size = m_size[0];
            for (uint8_t i = 1; i < N; ++i)
            {
                size *= m_size[i];
            }
            return size;
        }

        bool isContinuous() const
        {
            if (m_stride[N - 1] == 1)
            {
                for (int16_t i = N - 2; i >= 0; --i)
                {
                    const auto prev_stride = m_stride[i + 1];
                    const auto prev_shape = m_size[i + 1];
                    if (m_stride[i] != static_cast<int32_t>(prev_stride * prev_shape))
                    {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        uint8_t numDimensions() const { return N; }
    };

    template <>
    class Shape<0>
    {
      public:
        template <class... T>
        size_t index(T&&... args) const
        {
            return 0;
        }

        MT_XINLINE uint32_t operator[](int16_t) const { return 0; }
        MT_XINLINE uint32_t getStride(int16_t) const { return 1; }

        void setShape(uint8_t, uint32_t) {}
        void setStride(uint8_t, uint32_t) {}
        bool operator==(const Shape&) const { return true; }

        size_t numElements() const { return 1; }
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

    template <uint32_t NUMERATOR, uint32_t DENOMINATOR, uint8_t N>
    Shape<N> copyScaled(const Shape<N>& shape)
    {
        Shape<N> out_shape;
        for (uint8_t i = 0; i < N; ++i)
        {
            out_shape.setShape(i, (NUMERATOR * shape[i]) / DENOMINATOR);
            out_shape.setStride(i, (NUMERATOR * shape.getStride(i)) / DENOMINATOR);
        }
        return out_shape;
    }

    template <uint8_t D>
    Shape<D - 1> squeezeDim(uint8_t dim, const Shape<D>& shape)
    {
        Shape<D - 1> out_shape;

        uint8_t j = 0;
        for (uint8_t i = 0; i < D; ++i)
        {
            if (i != dim)
            {
                out_shape.setShape(j, shape[i]);
                out_shape.setStride(j, shape.getStride(i));
                ++j;
            }
        }
        return out_shape;
    }

    template <uint8_t N>
    void unsqueeze(const Shape<N>& in, Shape<N + 1>& out, uint8_t dim)
    {
        // TODO copy stride
        uint8_t out_dim = 0;
        for (uint8_t i = 0; i < N; ++i)
        {
            out.setShape(out_dim, in[i]);
            if (i == dim)
            {
                out.setShape(out_dim + 1, 1);
                ++out_dim;
            }
            ++out_dim;
        }
        out.calculateStride();
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
