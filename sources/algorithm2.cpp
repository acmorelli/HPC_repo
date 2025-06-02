#include <algorithm>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <cstring>
#include "algorithms.h"

/**
 * optimized circulant allgather-merge (assignment algorithm 2)
 * - handles non-power-of-two process counts correctly
 * - uses fixed-size buffers to avoid dynamic reallocation
 * - merges using std::merge and std::inplace_merge as needed
 */

int HPC_AllgatherMergeCirculant(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    using dtype = tuwtype_t;
    const dtype* send = static_cast<const dtype*>(sendbuf);
    dtype* recv = static_cast<dtype*>(recvbuf);

    // if there's only one process, we just copy the input to the output
    if (p == 1) {
        std::copy(send, send + sendcount, recv);
        return MPI_SUCCESS;
    }

    // we calculate how many rounds we need based on log2(p)
    int q = std::ceil(std::log2(p));

    // s stores the skip sizes for each round
    std::vector<int> s(q + 1);
    s[q] = p;
    for (int k = q - 1; k >= 0; --k)
        s[k] = (s[k + 1] + 1) / 2;

    // we preallocate space for the merged data and a temporary buffer
    std::vector<dtype> M(sendcount * p);
    std::vector<dtype> T(sendcount * p);
    int m_active = 0; // keeps track of how many elements are valid in M

    for (int k = 0; k < q; ++k) {
        int sk = s[k];
        int sk1 = s[k + 1];
        int epsilon = sk1 & 1;

        // we calculate where to send to and receive from
        int send_to = (rank - sk + epsilon + p) % p;
        int recv_from = (rank + sk - epsilon) % p;

        if (epsilon == 0) {
            if (k == 0) {
                // in the first step we send our local input and receive into M
                MPI_Sendrecv(send, sendcount, sendtype, send_to, 0,
                             M.data(), sendcount, recvtype, recv_from, 0,
                             comm, MPI_STATUS_IGNORE);
                m_active = sendcount;
            } else {
                // from second round onwards, we merge local input with M into T
                std::merge(send, send + sendcount,
                           M.begin(), M.begin() + m_active,
                           T.begin());
                int merged = sendcount + m_active;

                // we send T and receive the same amount into M after m_active
                MPI_Sendrecv(T.data(), merged, sendtype, send_to, 0,
                             M.data() + m_active, merged, recvtype, recv_from, 0,
                             comm, MPI_STATUS_IGNORE);

                // now we merge everything back into M
                std::inplace_merge(M.begin(), M.begin() + m_active,
                                   M.begin() + m_active + merged);
                m_active += merged;
            }
        } else {
            // for epsilon == 1, we just send and receive M directly
            MPI_Sendrecv(M.data(), m_active, sendtype, send_to, 0,
                         M.data() + m_active, m_active, recvtype, recv_from, 0,
                         comm, MPI_STATUS_IGNORE);

            // we merge both parts in-place and double the active size
            std::inplace_merge(M.begin(), M.begin() + m_active,
                               M.begin() + 2 * m_active);
            m_active *= 2;
        }
    }

    // finally, we merge our original input with everything we received into recv
    std::merge(send, send + sendcount,
               M.begin(), M.begin() + m_active,
               recv);

    return MPI_SUCCESS;
}
