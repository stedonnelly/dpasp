#ifndef _PASP_COPTIMIZE
#define _PASP_COPTIMIZE

#include <stdbool.h>
#include <stdlib.h>

/* Threshold on m (number of credal facts) above which coordinate ascent (bfca) is used instead of
 * exhaustive brute-force (bf). bf is O(2^m) and exact; bfca is O(m * tries) and approximate.
 * When bf is used and num_threads > 1, the corner enumeration is parallelized across threads. */
#define BFCA_THRESHOLD 20
#define BFCA_TRIES(m) ((m) * 10)

#define BFCA_MAXIMIZE -1
#define BFCA_MINIMIZE 1

double f(double *X, bool *S, double *C, size_t n, size_t m);

double bfca(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, int maxmin, size_t tries, bool smp);

void bf(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp);

void bf_minmax(double *X, bool *S_a, bool *S_b, bool* S_c, bool* S_d, double *C_a,
    double *C_b, double *C_c, double *C_d, double *L, double *U, size_t n_a, size_t n_b,
    size_t n_c, size_t n_d, size_t m, double *low, double *up);

/* Adaptive optimization: uses parallel bf for exact, bfca for approximate when m > threshold. */
void optimize_credal(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L,
    double *U, size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp,
    size_t num_threads);

void optimize_credal_minmax(double *X, bool *S_a, bool *S_b, bool *S_c, bool *S_d,
    double *C_a, double *C_b, double *C_c, double *C_d, double *L, double *U,
    size_t n_a, size_t n_b, size_t n_c, size_t n_d, size_t m, double *low, double *up,
    size_t num_threads);

#endif
