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
#include <string.h>
#include <stdlib.h>

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
int totalNumLoops = 3;
// WARNING: if we want to check more than once, and use a total number of loops
// greater than 2, we **need** to set checkEvery to 1. This is because, if we
// use a user provided value (and we modify the global variable), this changes
// won't be persisted across migrations.
int checkEvery = 1;
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
    // bool mustCheck = nLoops == totalNumLoops;

    if (globalArgc == -1 || globalArgv == nullptr) {
        printf("Error! LAMMPS is a nullptr!\n");
    }
    MPI_Init(&globalArgc, &globalArgv);

    // Get rank
    int rank;
    int worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    for (int i = 0; i < nLoops; i++) {
        // Barrier to make sure all ranks are in sync (including those that
        // have been migrated)
        printf("Rank %i/%i executing loop %i\%i\n", rank, worldSize, i + 1, nLoops);
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("Rank %i/%i exitting first barrier!\n", rank, worldSize);

        lammps = new LAMMPS(globalArgc, globalArgv, MPI_COMM_WORLD);
        doLammps();
        // if (mustCheck && ((i + 1)  % checkEvery == 0) && ((i + 1) != totalNumLoops)) {
        // Check for migration opportunities if this iteration is a multiple of
        // check every, and it is not the last iteration. Bear in mind that the
        // meaning of "last" iteration will vary between migrated and non-
        // migrated ranks
        if (((i + 1)  % checkEvery == 0) && ((i + 1) != nLoops)) {
#ifdef __faasm
            if (rank == 0) {
                printf("---------------------------------------------------\n");
                printf("LAMMPS-Migrate checking for migration opportunities\n");
                printf("---------------------------------------------------\n");
            }
#endif
            printf("Rank %i/%i in mig. check branch (iter: %i\%i)\n", rank, worldSize, i + 1, nLoops);
            // printf("Rank %i/%i entering second barrier!\n", rank, worldSize);
            MPI_Barrier(MPI_COMM_WORLD);
            // printf("Rank %i/%i entering second barrier!\n", rank, worldSize);
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

#ifdef __faasm
    long inputSize = faasmGetInputSize();
    uint8_t* inputBuffer = (uint8_t*) malloc(inputSize * sizeof(uint8_t));
    faasmGetInput(inputBuffer, inputSize);

    char* inputStr = (char*) inputBuffer;
    int checkEveryIn = atoi(strtok(inputStr, " "));
    int numLoopsIn = atoi(strtok(NULL, " "));

    // Filthy hack to set the check period without modifying the function
    // signature. Note that the migrated functions won't see the updated
    // value as we don't migrate global variables, but that's OK as we don't
    // support migrating a function twice.
    int* numTotalLoopsPtr = &totalNumLoops;
    *numTotalLoopsPtr = numLoopsIn;
    int* checkEveryPtr = &checkEvery;
    *checkEveryPtr = checkEveryIn;
#endif
    printf(
      "Starting MPI migration v2 checking at iter %i/%i\n", checkEvery, totalNumLoops);

    doBenchmark(totalNumLoops);
}
