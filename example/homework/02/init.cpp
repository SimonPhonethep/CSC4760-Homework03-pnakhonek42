using namespace std;
#include <Kokkos_Core.hpp>
#include <iostream>
#include <assert.h>
#include <string>
#include <mpi.h>
#include <cstdio>
#define GLOBAL_PRINT

class Domain
{
public:
  Domain(int _M, int _N, const char *_name="") : domain(new char[(_M+2)*(_N+2)]), M(_M), N(_N), name(_name)  {}
  virtual ~Domain() {delete[] domain;}
  char &operator()(int i, int j) {return domain[i*N+j];}
  char operator()(int i, int j)const {return domain[i*N+j];}

  int rows() const {return M;}
  int cols() const {return N;}

  const string & myname() const {return name;}

  char *rawptr() {return domain;}
  
protected:
  Kokkos::View<char**> domain; 
  int M;
  int N;

  string name;
};

void zero_domain(Domain &domain);
void print_domain(Domain &domain, int rank);
void update_domain(Domain &new_domain, Domain &old_domain, int size, int myrank, MPI_Comm comm);
void parallel_code(int M, int N, int iterations, int size, int myrank, MPI_Comm comm);

int main(int argc, char **argv)
{

  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  int M, N;
  int iterations;

  if(argc < 4)
  {
    cout << "usage: " << argv[0] << " M N iterations" << endl;
	Kokkos::finalize();
	MPI_Finalize();
    exit(0);
  }

  M = atoi(argv[1]);
  N = atoi(argv[2]);
  iterations = atoi(argv[3]);

  int size, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int array[3];
  if(myrank == 0)
  {
     M = atoi(argv[1]); N = atoi(argv[2]); iterations = atoi(argv[3]);

     array[0] = M;
     array[1] = N;
     array[2] = iterations;
     
  }
  MPI_Bcast(array, 3, MPI_INT, 0, MPI_COMM_WORLD);
  if(myrank != 0)
  {
    M = array[0];
    N = array[1];
    iterations = array[2];
  }

  parallel_code(M, N, iterations, size, myrank, MPI_COMM_WORLD);
  
  Kokkos::finalize();
  MPI_Finalize();
}

void parallel_code(int M, int N, int iterations, int size, int myrank, MPI_Comm comm)
{
  int nominal = M / size; int extra = M % size;
  int m = (myrank < extra) ? (nominal+1) : nominal;
  int n = N;
  
  Domain even_domain(m,n,"even Domain");
  Domain odd_domain(m,n,"odd Domain");

  zero_domain(even_domain);
  zero_domain(odd_domain);

#ifdef GLOBAL_PRINT
  Domain *global_domain = nullptr;
  int *thecounts = nullptr;
  int *displs = nullptr;

  if(0 == myrank)
  {
    thecounts  = new int[size];
    displs = new int[size];
    for(int i = 0; i < size; ++i)
      thecounts[i] = nominal*n; 
    for(int i = 0; i < extra; ++i)
      thecounts[i] += n;

    int count = displs[0] = 0;
    for(int i = 1; i < size; ++i)
    {
      count += thecounts[i-1];
      displs[i] = count;
    }
    
    global_domain = new Domain(M,N,"Global Domain");
    zero_domain(*global_domain);

    if((n >= 8) && (m >= 10))
    {
      (*global_domain)(0,(n-1)) = 1;
      (*global_domain)(0,0)     = 1;
      (*global_domain)(0,1)     = 1;

      (*global_domain)(8,5)     = 1;
      (*global_domain)(8,6)     = 1;
      (*global_domain)(8,7)     = 1;
      (*global_domain)(7,7)     = 1;
      (*global_domain)(6,6)     = 1;
    }
  }
  
  MPI_Scatterv(((myrank==0) ? global_domain->rawptr() : nullptr),
	       thecounts, displs, MPI_CHAR,
	       even_domain.rawptr(), m*n, MPI_CHAR, 0, comm);
  
#else  
  if((n >= 8) && (m >= 10))
  {
#if 0    
    even_domain(0,(n-1)) = 1;
    even_domain(0,0)     = 1;
    even_domain(0,1)     = 1;
    
    even_domain(3,5) = 1;
    even_domain(3,6) = 1;
    even_domain(3,7) = 1;

    even_domain(6,7) = 1;
    even_domain(7,7) = 1;
    even_domain(8,7) = 1;
    even_domain(9,7) = 1;
#else
    // blinker at top left, touching right...
    even_domain(0,(n-1)) = 1;
    even_domain(0,0)     = 1;
    even_domain(0,1)     = 1;

    // and a glider:
    even_domain(8,5)     = 1;
    even_domain(8,6)     = 1;
    even_domain(8,7)     = 1;
    even_domain(7,7)     = 1;
    even_domain(6,6)     = 1;
#endif    
  }
#endif  

#ifdef GLOBAL_PRINT
    if(0 == myrank)
    {  
      cout << "Initial State:" << endl;
      print_domain(*global_domain, myrank);
    }
#else
    cout << "Initial State:" << i << endl;
    print_domain(*even, myrank);
#endif  

  Domain *odd, *even;
  odd = &odd_domain;
  even = &even_domain;

  for(int i = 0; i < iterations; ++i)
  {
    update_domain(*odd, *even, size, myrank, comm);

#ifdef GLOBAL_PRINT
    if(0 == myrank)
      cout << "Iteration #" << i << endl;
    MPI_Gatherv(odd->rawptr(), m*n, MPI_CHAR,
		((myrank==0) ? global_domain->rawptr() : nullptr),
		thecounts, displs, MPI_CHAR, 0, comm);

    if(0 == myrank)
      print_domain(*global_domain, myrank);
#else
    cout << "Iteration #" << i << endl; print_domain(*odd, myrank);
#endif

    // swap pointers:
    Domain *temp = odd;
    odd  = even;
    even = temp;
  }
#ifdef GLOBAL_PRINT
  if(0 == myrank)
  {
    delete global_domain;
    delete[] thecounts;
    delete[] displs;
  }
#endif  
}

void zero_domain(Domain &domain)
{
  Kokkos::deep_copy(domain.domain, char(0));
}

void print_domain(Domain &domain, int rank)
{
  cout << rank << ": " << domain.myname() << ":" <<endl;
  for(int i = 0; i < domain.rows(); ++i)
  {
    for(int j = 0; j < domain.cols(); ++j)
      cout << (domain(i,j) ? "*" : ".");
    cout << endl;
  }
}
inline char update_the_cell(char cell, int neighbor_count)
{
  char newcell;
  if(cell == 0) // dead now
    newcell = (neighbor_count == 3) ? 1 : 0;
  else // was live, what about now?
    newcell = ((neighbor_count == 2)||(neighbor_count == 3)) ? 1 : 0;
									    return newcell;
}
      
void update_domain(Domain &new_domain, Domain &old_domain, int size, int myrank, MPI_Comm comm) 
{
  Kokkos::parallel_for("update_domain", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {new_domain.rows()+1, new_domain.cols()+1}), KOKKOS_LAMBDA(int i, int j) {
    int neighbor_count = 0;
    for(int di = -1; di <= 1; ++di) {
      for(int dj = -1; dj <= 1; ++dj) {
        if(di == 0 && dj == 0) continue;
        neighbor_count += old_domain(i+di, j+dj);
      }
    }
    new_domain(i,j) = (old_domain(i,j) == 1 && (neighbor_count == 2 || neighbor_count == 3)) || (old_domain(i,j) == 0 && neighbor_count == 3);
  });
  Kokkos::fence(); 
}


