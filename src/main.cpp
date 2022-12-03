/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "lammps.h"
#include "input.h"

#include <mpi.h>
#include <cstdlib>
#include <stdio.h>

#if defined(LAMMPS_TRAP_FPE) && defined(_GNU_SOURCE)
#include <fenv.h>
#endif

#if defined(LAMMPS_EXCEPTIONS)
#include "exceptions.h"
#endif

#ifdef __faasm
#include <faasm/faasm.h>
#include <faasm/migrate.h>
#endif

using namespace LAMMPS_NS;

int globalArgc = -1;
char** globalArgv = nullptr;
int totalNumLoops = 2;
LAMMPS* lammps = nullptr;

void doLammps()
{
    if (lammps != nullptr) {
        lammps->input->file();
    } else {
        printf("Error! LAMMPS is a nullptr!\n");
    }
}

void doBenchmark(int nLoops)
{
    bool mustCheck = nLoops == totalNumLoops;

    if (globalArgc == -1 || globalArgv == nullptr) {
        printf("Error! LAMMPS is a nullptr!\n");
    }
    MPI_Init(&globalArgc, &globalArgv);

    // Get rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < nLoops; i++) {
        lammps = new LAMMPS(globalArgc, globalArgv, MPI_COMM_WORLD);
        doLammps();
        if (mustCheck && i == 0) {
#ifdef __faasm
            if (rank == 0) {
                printf("---------------------------------------------------\n");
                printf("LAMMPS-Migrate checking for migration opportunities\n");
                printf("---------------------------------------------------\n");
            }
#endif
            MPI_Barrier(MPI_COMM_WORLD);
#ifdef __faasm
            __faasm_migrate_point(&doBenchmark, (nLoops - i - 1));
#endif
        }
        delete lammps;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

int main(int argc, char **argv)
{
    // Persist argc and argv as global variables to not change the signature
    // of the `doBenchmark` method
    globalArgv = argv;
    globalArgc = argc;

    doBenchmark(2);
}
