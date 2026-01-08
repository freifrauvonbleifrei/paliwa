# SPDX-FileCopyrightText: Copyright Spack Project Developers.
# See COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# (cf. https://github.com/dev-build-deploy/reuse-me/issues/189)

from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class Paliwa(CMakePackage):
    """paliwa is a library for parallel lifting wavelet transforms."""

    git = "https://github.com/freifrauvonbleifrei/paliwa.git"

    maintainers("freifrauvonbleifrei")

    license("LGPL-3.0-or-later", checked_by="freifrauvonbleifrei")

    version("main", branch="main")

    variant("mpi", default=True, description="Enable MPI support")
    variant("tests", default=False, description="Build tests")

    depends_on("cxx", type="build")
    depends_on("cmake@3.25:", type="build")
    depends_on("ddc~splines~fft")
    depends_on("googletest", when="+tests", type="test")
    depends_on("kokkos")
    depends_on("mpi", when="+mpi")

    def cmake_args(self):
        args = [
            self.define("PALIWA_BUILD_EXAMPLES", False),
            self.define_from_variant("PALIWA_BUILD_TESTS", "tests"),
            self.define_from_variant("PALIWA_WITH_MPI", "mpi"),
        ]
        return args
