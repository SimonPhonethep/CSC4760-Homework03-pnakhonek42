#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char *argv[]) {
    int M = 25;

    int P, Q;
    int i, j;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    P = 1;
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

    if (coords[1] == 0) {
        if (coords[0] == 0) {
            int *x = (int *)malloc(M * sizeof(int));
            for (i = 0; i < M; i++) {
                x[i] = i;
            }
            int *local_x = (int *)malloc(M/P * sizeof(int));
            MPI_Scatter(x, M/P, MPI_INT, local_x, M/P, MPI_INT, 0, MPI_COMM_WORLD);
            free(x);
        } else {

            int *local_x = (int *)malloc(M/P * sizeof(int));
            MPI_Scatter(NULL, M/P, MPI_INT, local_x, M/P, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    int *y = (int *)malloc(M/P * sizeof(int));

    MPI_Bcast(y, M/P, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, y, M/P, MPI_INT, MPI_SUM, cart_comm);

    for (i = 0; i < P; i++) {
        if (coords[0] == i) {
            printf("Process (%d, %d) - y: ", coords[0], coords[1]);
            for (j = 0; j < M/P; j++) {
                printf("%d ", y[j]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    free(y);

    return 0;
}