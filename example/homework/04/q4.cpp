#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char *argv[]) {
    int M = 25;

    int P = 1;
    int Q;
    int i;
    int j;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Q = size;

    if (rank == 0) {
        printf("P = %d, Q = %d\n", P, Q);
    }

    int dims[2] = {P, Q};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int *x = (int *)malloc(M * sizeof(int));
    if (coords[1] == 0) {
        int *local_x = (int *)malloc(M / P * sizeof(int));
        if (coords[0] == 0) {
            for (i = 0; i < M; i++) {
                x[i] = i; 
            }
            MPI_Scatter(x, M / P, MPI_INT, local_x, M / P, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(NULL, M / P, MPI_INT, local_x, M / P, MPI_INT, 0, MPI_COMM_WORLD);
        }
        free(local_x);
    }

    int *y = (int *)malloc(M * sizeof(int));
    if (coords[0] == 0) {
        if (coords[1] == 0) {
            for (i = 0; i < M; i++) {
                y[i] = i * size + rank; 
            }
        }
        MPI_Scatter(y, M / P, MPI_INT, y, M / P, MPI_INT, 0, cart_comm);
    }
    free(y);

    MPI_Bcast(x, M, MPI_INT, 0, cart_comm);

    int local_dot_product = 0;
    int global_dot_product = 0;
    for (i = 0; i < M / P; i++) {
        local_dot_product += x[i] * y[i];
    }

    MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global dot product: %d\n", global_dot_product);
    }

    MPI_Finalize();
    free(x);

    return 0;
}