#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int M = 25;

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int P = size; 
    int Q = 1;

    MPI_Comm split_comm1;
    int color1 = rank / Q;
    MPI_Comm_split(MPI_COMM_WORLD, color1, rank, &split_comm1);

    MPI_Comm split_comm2;
    int color2 = rank % Q;
    MPI_Comm_split(MPI_COMM_WORLD, color2, rank, &split_comm2);

    double* x = (double*) malloc(M * sizeof(double));
    double* y = (double*) malloc(M * sizeof(double));

    if (rank == 0) {
        for (int i = 0; i < M; i++) {
            x[i] = i;
        }
    }

    MPI_Scatter(x, M/Q, MPI_DOUBLE, x, M/Q, MPI_DOUBLE, 0, split_comm1);

    MPI_Bcast(x, M/Q, MPI_DOUBLE, 0, split_comm2);

    MPI_Allgather(x, M/Q, MPI_DOUBLE, y, M/Q, MPI_DOUBLE, split_comm2);

    free(x);
    free(y);
    MPI_Comm_free(&split_comm1);
    MPI_Comm_free(&split_comm2);

    MPI_Finalize();
    return 0;
}
