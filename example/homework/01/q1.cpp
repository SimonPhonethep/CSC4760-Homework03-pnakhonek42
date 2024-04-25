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


    MPI_Comm grid_comm;
    int dims[2] = {P, Q};
    int periods[2] = {0, 0}; 
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, coords[0], rank, &col_comm); 
    MPI_Comm_split(grid_comm, coords[1], rank, &row_comm); 

    double* x = (double*) malloc(M * sizeof(double));
    double* y = (double*) malloc(M * sizeof(double));

    if (rank == 0) {
        for (int i = 0; i < M; i++) {
            x[i] = i; 
        }
    }

    if (coords[1] == 0) {
        MPI_Scatter(coords[0] == 0 ? x : NULL, M / P, MPI_DOUBLE, x, M / P, MPI_DOUBLE, 0, col_comm);
    }

    MPI_Bcast(x, M / P, MPI_DOUBLE, 0, row_comm);
    MPI_Allgather(x, M / P, MPI_DOUBLE, y, M / P, MPI_DOUBLE, row_comm);

    free(x);
    free(y);

    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&grid_comm);

    MPI_Finalize();
    return 0;
}
