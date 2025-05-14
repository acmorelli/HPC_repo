// cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
  // or for debugging:
// cmake -DCMAKE_BUILD_TYPE=Debug
// cmake --build build --parallel
// mpirun -np 4 ./build/allgather_merge -m 10 -a 0 -t 0 -c

// -m: message size per process
// -a: 0 is algorithm 0 (baseline)
// -t: 0 input type 0 (all values=rank)
// -c: check result correctness  


// mpirun -np 4 ./build/allgather_merge -a 0
  // runs all msg sizes and input types. output in latex table

// clean rebuild
"""
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
"""

#include <mpi.h>
#include "algorithms.h"
#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>  // for memcpy
#include "algorithms.h"

using tuwtype_t = int;

// Min-heap item for p-way merge
"""lets the heap know how to compare elements based on value"""
struct HeapItem {
    tuwtype_t value; //value inside one of the arrays
    int block_idx;  //identifies which block (allgather took place already)
    int value_idx; //idx inside the block

    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// Optimized p-way merge (defined in the same file)
inline void p_way_merge_optimized(const tuwtype_t* input, tuwtype_t* output,
                                  int p, int block_size) {
    """min heap: extracts the smallest number everytime"""
    using MinHeap = std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<>>;

    MinHeap heap;

    // Stack-allocated index tracking
    int* indices = static_cast<int*>(alloca(p * sizeof(int)));

    for (int i = 0; i < p; ++i) {
        indices[i] = 0;
        if (block_size > 0) {
            const tuwtype_t* block = input + i * block_size;
            heap.push({block[0], i, 0});
        }
    }

    int out_idx = 0;
    while (!heap.empty()) {
        HeapItem top = heap.top(); heap.pop();
        output[out_idx++] = top.value;

        int next_offset = top.value_idx + 1;
        if (next_offset < block_size) {
            const tuwtype_t* block = input + top.block_idx * block_size;
            heap.push({block[next_offset], top.block_idx, next_offset});
        }
    }
}

// implementation
int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const tuwtype_t* sbuf = static_cast<const tuwtype_t*>(sendbuf);
    tuwtype_t* rbuf = static_cast<tuwtype_t*>(recvbuf);
    int total_count = sendcount * size;

    // MPI_IN_PLACE avoids copying data locally (might or might not help)
    // sendbuf and recvbuf are the same to avoid duplication in memory
    // so skipping a local copy can save overhead
    if (sendbuf == MPI_IN_PLACE) {
        MPI_Allgather(MPI_IN_PLACE, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    } else {
        MPI_Allgather(sbuf, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    }

    // create temporary buffer for merging
    """
    exactly one buffer to hold final sorted result

    malloc instead of std::vector or new[] avoids constructor/destructor overhead

    single temporary avoids reallocating/resizing when we merge

    we avoid overwriting recvbuf while still reading from it
    """
    tuwtype_t* merged = static_cast<tuwtype_t*>(malloc(total_count * sizeof(tuwtype_t)));

    // merge the p sorted blocks from recvbuf into merged
    p_way_merge_optimized(rbuf, merged, size, sendcount);

    // copy result back to recvbuf - we overwrite only once
    memcpy(rbuf, merged, total_count * sizeof(tuwtype_t));
    free(merged);

    return MPI_SUCCESS;
}

