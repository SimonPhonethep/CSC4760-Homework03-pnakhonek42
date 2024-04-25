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
    int Q = size;

    MPI_Comm split_comm1;
    int color1 = rank / Q;
    MPI_Comm_split(MPI_COMM_WORLD, color1, rank, &split_comm1);

    MPI_Comm split_comm2;
    int color2 = rank % Q;
    MPI_Comm_split(MPI_COMM_WORLD, color2, rank, &split_comm2);

    int elements_p_process = M / Q; 
  
    double* x = (double*) malloc(M * sizeof(double));
    double* y = (double*) malloc(M * sizeof(double));
    double* new_y = (double*) malloc(elements_p_process * sizeof(double));

    if (rank == 0) {
        for (int i = 0; i < M; i++) {
            x[i] = i;
        }
    }

    MPI_Scatter(x, M/Q, MPI_DOUBLE, x, M/Q, MPI_DOUBLE, 0, split_comm1);

    MPI_Bcast(x, M/Q, MPI_DOUBLE, 0, split_comm2);

    MPI_Allgather(x, M/Q, MPI_DOUBLE, y, M/Q, MPI_DOUBLE, split_comm2);

    for (int J = 0; J < M; J++) {
        int j = J / Q; 
        int q = J % Q; 
        if (q == rank) {
            new_y[j] = y[J];
        }
    }

    double dot_product = 0.0;
    for (int i = 0; i < elements_p_process; i++) {
        dot_product += x[i] * new_y[i];
    }

    double global_dot_product;
    MPI_Allreduce(&dot_product, &global_dot_product, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Dot product: %f\n", global_dot_product);
    }
    
    free(x);
    free(y);
    free(new_y);
    MPI_Comm_free(&split_comm1);
    MPI_Comm_free(&split_comm2);

    MPI_Finalize();
    return 0;
}