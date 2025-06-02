#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>  // for memcpy
#include <cstdlib>  // for malloc and free
#include <iostream> 
#include "algorithms.h" 
using tuwtype_t = int; // using a custom alias here so we can change the data type later 

// struct for each item in the heap during merging
// it stores the value, which block it's from, and its index inside that block
struct HeapItem {
    tuwtype_t value;
    int block_idx;
    int value_idx;

    // we define '>' so std::priority_queue with std::greater can build a min-heap
    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// performs a p-way merge using a min-heap (optimized version)
// input: pointer to all gathered data, output: where to store sorted data
inline void p_way_merge_optimized(const tuwtype_t* input, tuwtype_t* output,
                                  int p, int block_size) {
    using MinHeap = std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<>>;

    MinHeap heap;

    // stack-allocate an array to keep track of each block's current index
    int* indices = static_cast<int*>(alloca(p * sizeof(int)));

    // initialize each block's index and push the first element into the heap
    for (int i = 0; i < p; ++i) {
        indices[i] = 0;
        if (block_size > 0) {
            const tuwtype_t* block = input + i * block_size;
            heap.push({block[0], i, 0}); // push first element of each block
        }
    }

    int out_idx = 0;
    // we keep extracting the smallest item and pushing the next item from the same block
    while (!heap.empty()) {
        HeapItem top = heap.top();
        heap.pop();
        output[out_idx++] = top.value;

        int next_offset = top.value_idx + 1;
        // check if there are more elements in this block
        if (next_offset < block_size) {
            const tuwtype_t* block = input + top.block_idx * block_size;
            heap.push({block[next_offset], top.block_idx, next_offset});
        }
    }
}

// main function to perform allgather followed by a merge sort
int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank); // get current process rank
    MPI_Comm_size(comm, &size); // get total number of processes

    const tuwtype_t* sbuf = static_cast<const tuwtype_t*>(sendbuf);
    tuwtype_t* rbuf = static_cast<tuwtype_t*>(recvbuf);
    int total_count = sendcount * size;

    // if we use MPI_IN_PLACE, we avoid sending from a separate send buffer
    // this is just an optimization and might slightly reduce memory usage
    if (sendbuf == MPI_IN_PLACE) {
        MPI_Allgather(MPI_IN_PLACE, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    } else {
        MPI_Allgather(sbuf, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    }

    // we allocate a temp buffer to store the merged result
    // this helps avoid doing in-place merging which is more complex
    tuwtype_t* merged = static_cast<tuwtype_t*>(malloc(total_count * sizeof(tuwtype_t)));
    if (!merged) {
        std::cerr << "Memory allocation failed for merged buffer\n";
        return MPI_ERR_NO_MEM;
    }

    // now we merge all the sorted blocks received from other processes
    p_way_merge_optimized(rbuf, merged, size, sendcount);

    // after merging, we copy the result back into the original recv buffer
    // we do a full memcpy so that recvbuf has the final sorted result
    std::memcpy(rbuf, merged, total_count * sizeof(tuwtype_t));
    free(merged); // free the temporary buffer

    return MPI_SUCCESS;
}
