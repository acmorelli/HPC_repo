#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "algorithms.h"
#include <queue>

/**
 * Correct Bruck-style allgather-merge (Assignment Algorithm 1)
 * - Handles any number of processes (including non-powers-of-two)
 * - Corrects buffer size mismatches and ensures safety
 */int HPC_AllgatherMergeBruck(const void *sendbuf, int sendcount,
    MPI_Datatype sendtype, void *recvbuf,
    int /* recvcount */, MPI_Datatype recvtype,
    MPI_Comm comm) {
int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);

using dtype = tuwtype_t;
const dtype* local = static_cast<const dtype*>(sendbuf);
dtype* result = static_cast<dtype*>(recvbuf);
int total_count = sendcount * size;

// this buffer will store all received blocks from all processes
std::vector<dtype> buffer(total_count);

// we copy our own local input block into the correct position
std::memcpy(&buffer[rank * sendcount], local, sendcount * sizeof(dtype));

// this array keeps track of which blocks we already have
std::vector<char> has_block(size, 0);
has_block[rank] = 1;

// we compute the number of steps we need in the Bruck algorithm
int logp = static_cast<int>(std::ceil(std::log2(size)));

// this is the main loop of the Bruck communication
for (int k = 0; k < logp; ++k) {
int offset = 1 << k;
int send_to = (rank - offset + size) % size;
int recv_from = (rank + offset) % size;

// we figure out which blocks we need to send in this round
std::vector<int> send_blocks;
for (int i = 0; i < size; ++i) {
if (has_block[i]) {
int rel = (i - rank + size) % size;
if (rel >= offset && rel < 2 * offset) {
send_blocks.push_back(i);
}
}
}

// we gather those blocks into a contiguous send buffer
int send_len = static_cast<int>(send_blocks.size()) * sendcount;
std::vector<dtype> send_data(send_len);
for (size_t i = 0; i < send_blocks.size(); ++i) {
std::memcpy(&send_data[i * sendcount], &buffer[send_blocks[i] * sendcount], sendcount * sizeof(dtype));
}

// prepare a receive buffer of the same size
std::vector<dtype> recv_data(send_len);

// non-blocking send and receive
MPI_Request reqs[2];
MPI_Isend(send_data.data(), send_len, sendtype, send_to, 0, comm, &reqs[0]);
MPI_Irecv(recv_data.data(), send_len, recvtype, recv_from, 0, comm, &reqs[1]);
MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

// store the received blocks in their expected positions
for (size_t i = 0; i < send_blocks.size(); ++i) {
int global_idx = (send_blocks[i] + offset) % size;
if (!has_block[global_idx]) {
std::memcpy(&buffer[global_idx * sendcount], &recv_data[i * sendcount], sendcount * sizeof(dtype));
has_block[global_idx] = 1;
}
}
}

// for non-power-of-two process counts, some blocks might still be missing
// we use an allgather to figure out which process has which blocks
std::vector<char> all_has_block(size * size);
MPI_Allgather(has_block.data(), size, MPI_CHAR, all_has_block.data(), size, MPI_CHAR, comm);

// we do additional communication to fetch missing blocks
std::vector<MPI_Request> fix_reqs;

for (int i = 0; i < size; ++i) {
if (!has_block[i]) {
for (int src = 0; src < size; ++src) {
if (all_has_block[src * size + i]) {
fix_reqs.emplace_back();
MPI_Irecv(&buffer[i * sendcount], sendcount, recvtype, src, 1000 + i, comm, &fix_reqs.back());
break;
}
}
}
}

for (int i = 0; i < size; ++i) {
if (has_block[i]) {
for (int dst = 0; dst < size; ++dst) {
if (!all_has_block[dst * size + i]) {
fix_reqs.emplace_back();
MPI_Isend(&buffer[i * sendcount], sendcount, sendtype, dst, 1000 + i, comm, &fix_reqs.back());
}
}
}
}

// wait for all pending sends/receives to complete
if (!fix_reqs.empty()) {
MPI_Waitall(static_cast<int>(fix_reqs.size()), fix_reqs.data(), MPI_STATUSES_IGNORE);
}

// now we do the final p-way merge of all sorted blocks into the result buffer
std::vector<const dtype*> block_ptrs(size);
for (int i = 0; i < size; ++i) {
block_ptrs[i] = &buffer[i * sendcount];
}

// this is a min-heap to perform a k-way merge efficiently
struct HeapItem {
dtype value;
int block, idx;
bool operator>(const HeapItem& o) const { return value > o.value; }
};

std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<>> min_heap;
std::vector<int> indices(size, 0);

// initialize heap with the first element from each block
for (int i = 0; i < size; ++i) {
min_heap.push({block_ptrs[i][0], i, 0});
}

// we repeatedly pop the smallest element and push the next one from that block
for (int i = 0; i < total_count; ++i) {
HeapItem item = min_heap.top(); min_heap.pop();
result[i] = item.value;
int next_idx = item.idx + 1;
if (next_idx < sendcount) {
min_heap.push({block_ptrs[item.block][next_idx], item.block, next_idx});
}
}

return MPI_SUCCESS;
}