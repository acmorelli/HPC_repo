/*
cmake --build build --parallel
mpirun -np 5 ./build/allgather_merge -a 1 -c

*/

/* Placeholder code for the distributed allgather-merge project*/
/* (C) Jesper Larsson Traff, Sascha Hunold, Ioannis Vardas, HPC 2025 */

#include <algorithm>
#include <assert.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <optional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algorithms.h"

#define WARMUP 2 // You can adjust the warmup and repeat counts
#define REPEAT 10

// Do not change the following
#define MICRO 1000000.0
#define MAX_MSG_SIZE 10000001
#define MSG_SIZES 6
#define ALGORITHMS 3
#define INPUT_TYPES 3

// Structure to hold command line arguments
struct ProgramOptions {
  int msg_size = 0;
  int algorithm = 0;
  int input_type = 0;
  bool check = false;
};

// Global variables
std::optional<ProgramOptions> options;
constexpr int msg_sizes[MSG_SIZES] = {1, 10, 100, 1000, 10000, 100000};
constexpr char algorithms[ALGORITHMS][32] = {"Baseline", "Bruck", "Circulant"};
enum algos { Baseline = 0, Bruck = 1, Circulant = 2 };

int HPC_AllgatherMergeSort(void *sendbuf, void *recvbuf, const int count,
                           MPI_Datatype datatype, MPI_Comm comm) {
  int rank, size;
  int mpierr;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Perform the all-gather operation
  mpierr =
      MPI_Allgather(sendbuf, count, datatype, recvbuf, count, datatype, comm);
  if (mpierr != MPI_SUCCESS)
    return mpierr;

  // Early return for single process
  if (size == 1)
    return MPI_SUCCESS;

  // Simply sort the entire buffer
  tuwtype_t *data = static_cast<tuwtype_t *>(recvbuf);
  std::sort(data, data + (count * size));

  return MPI_SUCCESS;
}

// Parse and check command line arguments
std::optional<ProgramOptions> parseCommandLine(int argc, char *argv[],
                                               int rank) {
  ProgramOptions options;
  int opt;

  static struct option long_options[] = {{"msgsize", required_argument, 0, 'm'},
                                         {"algo", required_argument, 0, 'a'},
                                         {"type", required_argument, 0, 't'},
                                         {"help", no_argument, 0, 'h'},
                                         {"check", no_argument, 0, 'c'},
                                         {0, 0, 0, 0}};

  while ((opt = getopt_long(argc, argv, "m:a:t:hc", long_options, nullptr)) !=
         -1) {
    switch (opt) {
    case 'm':
      options.msg_size = std::stoi(optarg);
      if (options.msg_size < 1 || options.msg_size > MAX_MSG_SIZE) {
        if (rank == 0) {
          std::cerr << "Input Error: message size must be an integer "
                    << "between 1 and " << MAX_MSG_SIZE << std::endl;
        }
        return std::nullopt;
      }
      break;
    case 'a':
      options.algorithm = std::stoi(optarg);
      if (options.algorithm < 0 || options.algorithm > 2) {
        if (rank == 0) {
          std::cerr
              << "Input Error: algorithm must be an integer between 0 and 2"
              << std::endl;
        }
        return std::nullopt;
      }
      break;
    case 't':
      options.input_type = std::stoi(optarg);
      if (options.input_type < 0 || options.input_type > 2) {
        if (rank == 0) {
          std::cerr << "Input Error: type must be an integer between 0 and 2"
                    << std::endl;
        }
        return std::nullopt;
      }
      break;
    case 'h':
      if (rank == 0) {
        std::cout << "Usage: " << argv[0]
                  << " [-m <value>] [-a <value>] [-t <value>] [-c <value>]"
                  << std::endl;
        std::cout << "  -m, --msgsize: Message size (default: 10)" << std::endl;
        std::cout << "  -a, --algo: Algorithm (0-baseline, 1-Bruck, "
                     "2-Circulant) (default: 0)"
                  << std::endl;
        std::cout << "  -t, --type: Input type (0-2) (default: 0)" << std::endl;
        std::cout << "  -c, --check: Verify results (default: false)"
                  << std::endl;
      }
      return std::nullopt;
    case 'c':
      options.check = true;
      break;
    default:
      if (rank == 0) {
        std::cerr << "Usage: " << argv[0]
                  << " [-m <value>] [-a <value>] [-t <value>] [-c <value>]"
                  << std::endl;
      }
      return std::nullopt;
    }
  }

  return options;
}

void run_all_measurements(int rank, int size, int algo, int msg_index, int type,
                          MPI_Comm comm, bool multiple) {
  int r, t, i, m;
  double start, stop;
  tuwtype_t *sendbuf, *recvbuf;
  double runtime[REPEAT];

  if (multiple)
    m = msg_sizes[msg_index];
  else
    m = msg_index;

  sendbuf = (tuwtype_t *)malloc(m * sizeof(tuwtype_t));
  assert(sendbuf != NULL);
  recvbuf = (tuwtype_t *)malloc(size * m * sizeof(tuwtype_t));
  assert(recvbuf != NULL);

  switch (type) {
  case 0:
    for (i = 0; i < m; i++)
      sendbuf[i] = rank;
    break;
  case 1:
    for (i = 0; i < m; i++)
      sendbuf[i] = i;
    break;
  case 2:
    for (i = 0; i < m; i++)
      sendbuf[i] = size * i + rank;
    break;
  default:
    break;
    // cannot happen
  }

  // Not sure if we need this
  int total_size = m * size;
  for (i = 0; i < total_size; i++)
    recvbuf[i] = -1;

  // Debug print
#ifdef DEBUG
  std::cout << "Process " << rank << " local data\n";
  for (int i = 0; i < m; i++) {
    printf("%d, ", sendbuf[i]);
  }
  printf("\n");
#endif

  for (r = 0, t = 0; r < WARMUP + REPEAT; r++) {
    MPI_Barrier(comm);
    MPI_Barrier(comm);

    start = MPI_Wtime();

    switch (algo) {
    case Baseline:
      HPC_AllgatherMergeBase(sendbuf, m, TUW_TYPE, recvbuf, m, TUW_TYPE, comm);
      break;
    case Bruck:
      HPC_AllgatherMergeBruck(sendbuf, m, TUW_TYPE, recvbuf, m, TUW_TYPE, comm);
      break;
    case Circulant:
      HPC_AllgatherMergeCirculant(sendbuf, m, TUW_TYPE, recvbuf, m, TUW_TYPE,
                                  comm);
      break;
    default:
      break;
      // cannot happen
    }

    stop = MPI_Wtime();

    MPI_Barrier(comm);
    MPI_Barrier(comm);
    if (r < WARMUP)
      continue;
    runtime[t++] = stop - start;
  }
  MPI_Allreduce(MPI_IN_PLACE, runtime, t, MPI_DOUBLE, MPI_MAX, comm);

// Debug print
#ifdef DEBUG
  if (rank == 0) {
    std::cout << "Process Rank 0: Printing index and item in merged array\n";
    for (i = 0; i < m * size; i++) {
      printf("%d ", recvbuf[i]);
    }
    printf("\n");
  }
#endif

  if (options->check == false) {
    if (rank == 0) {
      double tuwavg, tuwmin;
      tuwavg = 0.0;
      tuwmin = runtime[0];
      for (t = 0; t < REPEAT; t++) {
        tuwavg += runtime[t];
        if (runtime[t] < tuwmin)
          tuwmin = runtime[t];
      }
      tuwavg /= REPEAT;
      if (multiple == true) {
        if (msg_index == std::size(msg_sizes) - 1) {
          std::cout << " & " << std::scientific << std::setprecision(2)
                    << tuwavg * MICRO << " \\\\\n";
        } else {
          std::cout << " & " << std::scientific << std::setprecision(2)
                    << tuwavg * MICRO;
        }
      } else {
        std::cout << "Average runtime of algorithm: " << algorithms[algo] << ", msg size: " << m << ", type: " << type << " for " << REPEAT
                  << " runs: " << std::scientific << std::setprecision(2)
                  << tuwavg * MICRO << " us\n";
      }
      std::cout.flush();
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (options->check) {
    tuwtype_t *sendbuf_copy = NULL;
    tuwtype_t *recvbuf_copy = NULL;
    sendbuf_copy = (tuwtype_t *)malloc(m * sizeof(tuwtype_t));
    assert(sendbuf_copy != NULL);
    memcpy(sendbuf_copy, sendbuf, m * sizeof(tuwtype_t));
    recvbuf_copy = (tuwtype_t *)malloc(size * m * sizeof(tuwtype_t));
    assert(recvbuf_copy != NULL);

    /* Perform a complete sort and compare the other algorithms against it */
    HPC_AllgatherMergeSort(sendbuf_copy, recvbuf_copy, m, TUW_TYPE,
                           MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "\nChecking correctness of algorithm: " << algorithms[algo]
                << ", msg size: " << m << ", type: " << type << "\n";
      std::cout.flush();
    }
    for (i = 0; i < m * size; i++) {
      if (recvbuf[i] != recvbuf_copy[i]) {
        std::cerr << "Error: Process " << rank << " at index " << i
                  << " has value " << recvbuf[i] << " instead of "
                  << recvbuf_copy[i] << std::endl;
        std::cerr.flush();
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    if (rank == 0) {
      std::cout << "All processes have the correct result\n";
      std::cout << "-------------------------------------\n";
      std::cout.flush();
    }
    free(sendbuf_copy);
    free(recvbuf_copy);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  free(sendbuf);
  free(recvbuf);
}

int main(int argc, char *argv[]) {
  int rank, size;
  int i, j, k;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Parse command line arguments
  options = parseCommandLine(argc, argv, rank);
  if (!options) {
    MPI_Finalize();
    return options ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  // if the message size is 0, we run all measurements
  if (options->msg_size == 0) {
    // only run selected algorithm
    int k = options->algorithm;
    if (rank == 0 && !options->check) {
      std::cout << "\\multirow{" << ALGORITHMS << "}{*}{" << algorithms[k] << "}\n";
      std::cout.flush();
    }
    for (i = 0; i < INPUT_TYPES; ++i) { // Types 0, 1, 2
      if (rank == 0 && !options->check) {
        std::cout << "& " << i << " ";
        std::cout.flush();
      }
      for (j = 0; j < static_cast<int>(std::size(msg_sizes)); ++j) {
        run_all_measurements(rank, size, k, j, i, MPI_COMM_WORLD, true);
      }
      if (rank == 0) {
        if (!options->check) {
          if (i == INPUT_TYPES - 1) {
            std::cout << "\\hline\n";
          } else {
            std::cout << "\\cline{" << INPUT_TYPES - 1 << "-"
                      << INPUT_TYPES - 1 + MSG_SIZES << "}\n";
          }
          std::cout.flush();
        }
      }
    }
  
  } else {
    run_all_measurements(rank, size, options->algorithm, options->msg_size,
                         options->input_type, MPI_COMM_WORLD, false);
  }

  MPI_Finalize();
  return 0;
}
