#pragma once

namespace paliwa {

template <typename T> // with T for example std::array<long int, dimensionality>
constexpr void iterate_hierarchical_subspaces(
    const T &nodal_level, T &tmp_level, size_t current_dim,
    const std::function<void(const T &)> &callback) {
  assert(tmp_level.size() == nodal_level.size());
  if (current_dim < nodal_level.size()) {
    for (tmp_level[current_dim] = 0;
         tmp_level[current_dim] <= nodal_level[current_dim];
         ++tmp_level[current_dim]) {
      iterate_hierarchical_subspaces(nodal_level, tmp_level, current_dim + 1,
                                     callback);
    }
  } else {
    callback(tmp_level);
  }
}

template <typename InstancesType>
constexpr void fence_all_instances(InstancesType const &instances) {
  for (auto const &instance : instances) {
    instance.fence();
  }
}

} // namespace paliwa