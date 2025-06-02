#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>  // for memcpy
#include <cstdlib>  // for malloc and free
#include <iostream> // for debugging (if needed)
#include "algorithms.h" 
using tuwtype_t = int;

// Min-heap item for p-way merge
// Lets the heap know how to compare elements based on value
struct HeapItem {
    tuwtype_t value; // Value inside one of the arrays
    int block_idx;   // Identifies which block (allgather took place already)
    int value_idx;   // Index inside the block

    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// p-way merge with minHeap
inline void p_way_merge_optimized(const tuwtype_t* input, tuwtype_t* output,
                                  int p, int block_size) {
    // Min heap: extracts the smallest number every time
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
        HeapItem top = heap.top();
        heap.pop();
        output[out_idx++] = top.value;

        int next_offset = top.value_idx + 1;
        if (next_offset < block_size) {
            const tuwtype_t* block = input + top.block_idx * block_size;
            heap.push({block[next_offset], top.block_idx, next_offset});
        }
    }
}

// Implementation
int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const tuwtype_t* sbuf = static_cast<const tuwtype_t*>(sendbuf);
    tuwtype_t* rbuf = static_cast<tuwtype_t*>(recvbuf);
    int total_count = sendcount * size;

    // MPI_IN_PLACE avoids copying data locally (might or might not help)
    if (sendbuf == MPI_IN_PLACE) {
        MPI_Allgather(MPI_IN_PLACE, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    } else {
        MPI_Allgather(sbuf, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    }

    // Create temporary buffer for merging
    // Exactly one buffer to hold the final sorted result
    tuwtype_t* merged = static_cast<tuwtype_t*>(malloc(total_count * sizeof(tuwtype_t)));
    if (!merged) {
        std::cerr << "Memory allocation failed for merged buffer\n";
        return MPI_ERR_NO_MEM;
    }

    // Merge the p sorted blocks from recvbuf into merged
    p_way_merge_optimized(rbuf, merged, size, sendcount);

    // Copy result back to recvbuf - we overwrite only once
    std::memcpy(rbuf, merged, total_count * sizeof(tuwtype_t));
    free(merged);

    return MPI_SUCCESS;
}