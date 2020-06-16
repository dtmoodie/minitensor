#ifndef MINITENSOR_UTILITIES_HPP
#define MINITENSOR_UTILITIES_HPP
#include <utility>

namespace mt
{
    template <size_t... I>
    struct IndexSequence
    {
        template <size_t N>
        using append = IndexSequence<I..., N>;

        template <size_t N>
        using preppend = IndexSequence<N, I...>;
    };

    template <size_t N, size_t START = 0, size_t IDX = N>
    struct MakeIndexSequence : MakeIndexSequence<N - 1, START, IDX - 1>
    {
        using Super_t = MakeIndexSequence<N - 1, START, IDX - 1>;
        using type = typename Super_t::type::template append<Super_t::NEXT>;
        static constexpr const size_t NEXT = Super_t::NEXT + 1;
    };

    template <size_t N, size_t START>
    struct MakeIndexSequence<N, START, 0>
    {
        using type = IndexSequence<START>;
        static constexpr const size_t NEXT = START + 1;
    };

    template <size_t N, size_t START = 0>
    using makeIndexSequence = typename MakeIndexSequence<N - 1, START>::type;

    template <class Construct, class Src, size_t... Indecies>
    Construct constructFromStaticSequence(const Src& src, IndexSequence<Indecies...>)
    {
        return Construct(src[Indecies]...);
    }
} // namespace mt

#endif // MINITENSOR_UTILITIES_HPP