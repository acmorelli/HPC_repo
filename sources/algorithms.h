#ifndef ALGORITHMS_H_
#define ALGORITHMS_H_

#include <mpi.h>

#define TUW_TYPE MPI_INT
typedef int tuwtype_t;

#ifdef __cplusplus
extern "C" {
#endif
// Baseline algorithm
int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm);

// Bruck-like algorithm: Algorithm 1 in project description
int HPC_AllgatherMergeBruck(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf, int recvcount,
                            MPI_Datatype recvtype, MPI_Comm comm);

// Allreduce-like algorithm: Algorithm 2 in the project description
int HPC_AllgatherMergeCirculant(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm);

#ifdef __cplusplus
}
#endif
#endif // ALGORITHMS_H_
