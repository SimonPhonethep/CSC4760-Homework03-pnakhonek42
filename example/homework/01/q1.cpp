#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int M = 25;
    int P = 1;
    int Q;
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

    int *local_x = malloc((M/P) * sizeof(int));
    if (coords[1] == 0) {
        if (rank == 0) {
            int *x = malloc(M * sizeof(int));
            for (int i = 0; i < M; i++) {
                x[i] = i;
            }
            MPI_Scatter(x, M/P, MPI_INT, local_x, M/P, MPI_INT, 0, MPI_COMM_WORLD);
            free(x);
        } else {
            MPI_Scatter(NULL, M/P, MPI_INT, local_x, M/P, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    int *y = malloc((M/P) * sizeof(int));
    if (rank == 0) {
        for (int i = 0; i < M/P; i++) {
            y[i] = local_x[i]; 
        }
    }
    MPI_Bcast(y, M/P, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, y, M/P, MPI_INT, MPI_SUM, cart_comm);

    for (int i = 0; i < P; i++) {
        if (coords[0] == i) {
            printf("Process (%d, %d) - y: ", coords[0], coords[1]);
            for (int j = 0; j < M/P; j++) {
                printf("%d ", y[j]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    free(local_x);
    free(y);

    return 0;
}
