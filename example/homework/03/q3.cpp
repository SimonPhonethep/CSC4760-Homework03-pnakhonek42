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

        int dims[2] = {P, Q};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int local_size = (M + Q - 1) / Q;  
    int *y = (int *)malloc(local_size * sizeof(int));

    int *global_y = NULL;
    if (rank == 0) {
        global_y = (int *)malloc(M * sizeof(int));
        for (int i = 0; i < M; i++) {
            global_y[i] = i;  
        }
    }

    for (int i = 0; i < local_size; i++) {
        int idx = rank + i * Q;
        if (idx < M) {  
            MPI_Scatter(global_y + idx, 1, MPI_INT, y + i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    printf("Process (%d, %d) - y:", coords[0], coords[1]);
    for (int i = 0; i < M/Q; i++) {
        printf(" %d", y[i]);
    }
    printf("\n");

    MPI_Finalize();
    free(y);


    return 0;
}
