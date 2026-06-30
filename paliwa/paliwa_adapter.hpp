// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <optional>
#include <utility>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

namespace paliwa
{

template <
        class ChunkSpanType,
        class ExecSpace = Kokkos::DefaultExecutionSpace>
class AdaptedData
{
private:
    using value_type = typename ChunkSpanType::element_type;
    using domain_type = typename ChunkSpanType::discrete_domain_type;

    using allocator_type = ddc::KokkosAllocator<
            value_type,
            typename ExecSpace::memory_space>;

    using chunk_type = ddc::Chunk<
            value_type,
            domain_type,
            allocator_type>;

    using span_type = decltype(std::declval<chunk_type>().span_view());

private:
    // True if a temporary buffer was allocated.
    bool m_needs_copy = false;

    // Temporary storage when the source memory space is not directly accessible.
    std::optional<chunk_type> m_chunk;

    // View passed to the algorithm.
    span_type m_span;

public:
    AdaptedData() = default;

    AdaptedData(AdaptedData const&) = delete;
    AdaptedData& operator=(AdaptedData const&) = delete;

    AdaptedData(AdaptedData&&) = default;
    AdaptedData& operator=(AdaptedData&&) = default;

    [[nodiscard]] span_type view() const noexcept
    {
        return m_span;
    }

private:
    template <class ES, class CST>
    friend auto adapter_in(CST const& src, ES instance);

    template <class ES, class CST>
    friend void adapter_out(
            CST& dst,
            AdaptedData<CST, ES> const& adapted,
            ES instance);
};


template <
        class ExecSpace = Kokkos::DefaultExecutionSpace,
        class ChunkSpanType>
[[nodiscard]]
auto adapter_in(
        ChunkSpanType const& src,
        ExecSpace instance = ExecSpace())
{
    using adapted_type = AdaptedData<ChunkSpanType, ExecSpace>;
    using value_type = typename ChunkSpanType::element_type;

    adapted_type adapted;

    constexpr bool accessible = Kokkos::SpaceAccessibility<
            typename ExecSpace::memory_space,
            typename ChunkSpanType::memory_space>::accessible;

    if constexpr (accessible) {
        // Memory is directly accessible: reuse the existing view.
        adapted.m_span = src;
    } else {
        // Allocate a temporary buffer and copy the data.
        adapted.m_needs_copy = true;

        adapted.m_chunk.emplace(
                "adapter_buffer",
                src.domain(),
                ddc::KokkosAllocator<
                        value_type,
                        typename ExecSpace::memory_space>());

        adapted.m_span = ddc::parallel_deepcopy(
                instance,
                *adapted.m_chunk,
                src);
    }

    return adapted;
}


template <
        class ExecSpace = Kokkos::DefaultExecutionSpace,
        class ChunkSpanType>
void adapter_out(
        ChunkSpanType& dst,
        AdaptedData<ChunkSpanType, ExecSpace> const& adapted,
        ExecSpace instance = ExecSpace())
{
    // Nothing to do if no temporary buffer was used.
    if (!adapted.m_needs_copy) {
        return;
    }

    ddc::parallel_deepcopy(
            instance,
            dst,
            *adapted.m_chunk);
}

} // namespace paliwa