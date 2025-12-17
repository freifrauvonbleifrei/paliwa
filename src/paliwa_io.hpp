#pragma once

#include <fstream>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

template <typename ChunkType>
void dump_chunk_span_to_binary_file(ChunkType const span,
                                    std::string const &filename) {
  auto host_mirror_view =
      ddc::create_mirror_view_and_copy(Kokkos::SharedHostPinnedSpace(), span);
  std::ofstream file(filename, std::ios::trunc | std::ios::binary);
  for (size_t i = 0; i < host_mirror_view.size(); ++i) {
    file.write(reinterpret_cast<char *>(&host_mirror_view.data_handle()[i]),
               sizeof(host_mirror_view.data_handle()[i]));
  }
}
