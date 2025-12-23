<!--
SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md

SPDX-License-Identifier: LGPL-2.1-or-later
-->
# paliwa : A code for parallel lifting wavelets

[![Build & Test CI](https://github.com/freifrauvonbleifrei/paliwa/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/freifrauvonbleifrei/paliwa/actions/workflows/build_and_test.yml/)
[![REUSE compliant](https://github.com/freifrauvonbleifrei/paliwa/actions/workflows/licenses.yaml/badge.svg)](https://github.com/freifrauvonbleifrei/paliwa/actions/workflows/licenses.yaml/)
![C++≥20](https://img.shields.io/badge/c++-%E2%89%A520-blue.svg)

## Why lifting wavelets?

Lifting wavelets are efficient hierarchical in-place transformations.
They can be useful when conservation of mass, velocity, energy,
and higher momenta of a function is required across scales.

Examples of lifting wavelets include (cf.
[Sweldens 1998](https://epubs.siam.org/doi/abs/10.1137/S0036141095289051)):

- biorthogonal / CDF wavelets
- [Cohen, Daubechies, Feauveau 1992](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.3160450502),
[Huber 2002](https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/1696)
- interpolets / "lazy wavelets" / "hierarchical basis" functions
[Griebel, Thurner 1995](https://doi.org/10.1108/EUM0000000004119)

Typical applications include multi-scale simulations and image compression.

## Why parallel?

In the context of higher dimensionalities, even lifting wavelet transforms
can take long, due to the curse of dimensionality.
But at each scale, lifting transforms have significant potential for parallelism,
which can be exploited by [DDC](https://github.com/CExA-project/ddc) and
[Kokkos](https://github.com/kokkos/kokkos/) on the shared-memory level.
If the function data becomes too large for a single node,
it can be further parallelized with MPI.

## Installation

First, follow the [instructions to set up a spack environment with ddc](https://ddc.mdls.fr/installation.html#autotoc_md38).
Activate the environment, then:

```bash
git clone https://github.com/freifrauvonbleifrei/paliwa.git
mkdir -p build
cmake -DPALIWA_BUILD_TESTS=ON -DPALIWA_WITH_MPI=ON -B ./build
cmake --build ./build/ --config Release
```

(check out the [CI workflow file](./.github/workflows/build_and_test.yml)
for details on a compatible environment.)

## Usage

Currently, the best usage documentation are in the [test codes](./test/).

After executing the test [`full_integration_2d`](https://github.com/freifrauvonbleifrei/paliwa/blob/main/test/test_paliwa_combination_technique.cpp#L366),
you can visualize its results by running

```bash
python3 ../src/postprocess.py
```

in the build folder.
The script creates a png file.
