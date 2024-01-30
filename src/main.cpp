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
int numNetLoops = -1;
LAMMPS* lammps = nullptr;

void doLammps()
{
    if (lammps != nullptr) {
        lammps->input->file();
    } else {
        printf("Error! LAMMPS is a nullptr!\n");
    }
}

void doAllToAll(int rank, int worldSize, int nLoops)
{
    int chunkSize = 2;
    int fullSize = worldSize * chunkSize;

    // Arrays for sending and receiving
    int sendBuf[fullSize];
    int expected[fullSize];
    int actual[fullSize];

    // Populate data
    for (int i = 0; i < fullSize; i++) {
        // Send buffer from this rank
        sendBuf[i] = (rank * 10) + i;

        // Work out which rank this chunk of the expectation will come from
        int rankOffset = (rank * chunkSize) + (i % chunkSize);
        int recvRank = i / chunkSize;
        expected[i] = (recvRank * 10) + rankOffset;
    }

    for (int i = 0; i < nLoops; i++) {
        MPI_Alltoall(
          sendBuf, chunkSize, MPI_INT, actual, chunkSize, MPI_INT, MPI_COMM_WORLD);
    }
}

void doBenchmark(int nLoops)
{
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
        printf("Rank %i/%i executing loop %i/%i\n", rank, worldSize, i + 1, nLoops);
        MPI_Barrier(MPI_COMM_WORLD);

        // Compute-intensive part of the benchmark
        lammps = new LAMMPS(globalArgc, globalArgv, MPI_COMM_WORLD);
        doLammps();

        // Network-intensive part of the benchmark
        doAllToAll(rank, worldSize, numNetLoops);

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
            printf("Rank %i/%i in mig. check branch (iter: %i/%i)\n", rank, worldSize, i + 1, nLoops);
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

    // To get the benchmark parameters, we use Faasm's input data, or env.
    // variables for native MPI (we must pass them using mpirun -x ENV -x ENV2)
#ifdef __faasm
    long inputSize = faasmGetInputSize();
    uint8_t* inputBuffer = (uint8_t*) malloc(inputSize * sizeof(uint8_t));
    faasmGetInput(inputBuffer, inputSize);

    // Faasm's configuration has three parameters:
    // - numLoops: how many iterations does the benchmark do (each iteration
    //             consists of a compute and a network bound part)
    // - checkEvery: how often do we check for migration opportunities
    // - numNetLoops: how many iterations do we do in the network-bound part
    //                of the main experiment loop
    //  We pass them separated by a space: "{} {} {}"
    char* inputStr = (char*) inputBuffer;
    int checkEveryIn = atoi(strtok(inputStr, " "));
    int numLoopsIn = atoi(strtok(NULL, " "));
    int numNetLoopsIn = atoi(strtok(NULL, " "));
#else
    char* inputStr = getenv("FAASM_BENCH_PARAMS");

    // For the native configuration, checkEvery is always equal to numLoops
    // (as we can never migrate native jobs), so we only need to pass two
    // values separated by a colon
    int numLoopsIn = atoi(strtok(inputStr, ":"));
    int checkEveryIn = numLoopsIn;
    int numNetLoopsIn = atoi(strtok(NULL, ":"));
#endif

    // Hacky way to pass the input data (not LAMMPS argc/arv) to the benchmark
    int* numTotalLoopsPtr = &totalNumLoops;
    *numTotalLoopsPtr = numLoopsIn;
    int* checkEveryPtr = &checkEvery;
    *checkEveryPtr = checkEveryIn;
    int* numNetLoopsPtr = &numNetLoops;
    *numNetLoopsPtr = numNetLoopsIn;

    printf(
      "Starting MPI migration v3.1 checking at iter %i/%i (%i net loops)\n", checkEvery, totalNumLoops, numNetLoops);

    doBenchmark(totalNumLoops);
}
