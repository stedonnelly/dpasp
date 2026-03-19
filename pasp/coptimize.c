#include "coptimize.h"
#include <pthread.h>

/* The polynomial to evaluate, where X are the variables, S are the signs of each factor, C are the
 * coefficients, n are the number of terms and m are the number of variables. For example, the
 * following polynomial
 *
 *   f(x, y, z) = 0.2*(1-x)*(1-y)*z+0.4*(1-x)*y*(1-z)+0.3*x*(1-y)*(1-z)+0.5*x*y*(1-z)
 *
 * under the evaluation f(0.2, 0.5, 0.7) would be represented as
 *
 *   X = {0.2, 0.5, 0.7}
 *   S = {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0}
 *   C = {0.2, 0.4, 0.3, 0.5}
 *   n = 4
 *   m = 3
 *
 * Note that |X| = m, |S| = n*m, and |C| = n.
 */
double f(double *X, bool *S, double *C, size_t n, size_t m) {
  size_t i, j, u;
  double s, y;
  for (i = s = 0; i < n; ++i) {
    y = 1;
    for (j = 0; j < m; ++j) {
      u = m*i+j;
      y *= S[u] ? X[j] : 1-X[j];
    }
    s += C[i]*y;
  }
  return s;
}

#define BFCA_MAX_M 64

/* ---- Parallel brute-force infrastructure ---- */

typedef struct {
  bool *S_a, *S_b;
  double *C_a, *C_b, *L, *U;
  size_t n_a, n_b, m;
  unsigned long long start, end;
  double low, up;
  bool smp;
} bf_task_t;

typedef struct {
  bool *S_a, *S_b, *S_c, *S_d;
  double *C_a, *C_b, *C_c, *C_d, *L, *U;
  size_t n_a, n_b, n_c, n_d, m;
  unsigned long long start, end;
  double low, up;
} bf_minmax_task_t;

static void *bf_worker(void *arg) {
  bf_task_t *t = (bf_task_t *)arg;
  double X[BFCA_MAX_M];
  size_t m = t->m;
  t->low = 1.0; t->up = 0.0;
  for (unsigned long long i = t->start; i < t->end; ++i) {
    for (size_t j = 0; j < m; ++j) X[j] = ((i >> j) & 1) ? t->L[j] : t->U[j];
    double a = f(X, t->S_a, t->C_a, t->n_a, m);
    double b = f(X, t->S_b, t->C_b, t->n_b, m);
    if (t->smp) {
      if (t->low > a) t->low = a;
      if (t->up < b) t->up = b;
    } else {
      double y = a + b;
      if (y != 0) y = a / y;
      if (t->low > y) t->low = y;
      if (t->up < y) t->up = y;
    }
  }
  return NULL;
}

static void *bf_minmax_worker(void *arg) {
  bf_minmax_task_t *t = (bf_minmax_task_t *)arg;
  double X[BFCA_MAX_M];
  size_t m = t->m;
  t->low = 1.0; t->up = 0.0;
  for (unsigned long long i = t->start; i < t->end; ++i) {
    for (size_t j = 0; j < m; ++j) X[j] = ((i >> j) & 1) ? t->L[j] : t->U[j];
    double a = f(X, t->S_a, t->C_a, t->n_a, m);
    double b = f(X, t->S_b, t->C_b, t->n_b, m);
    double c = f(X, t->S_c, t->C_c, t->n_c, m);
    double d = f(X, t->S_d, t->C_d, t->n_d, m);
    double y = a + d;
    if (y != 0) y = a / y;
    double z = b + c;
    if (z != 0) z = b / z;
    if (t->low > y) t->low = y;
    if (t->up < z) t->up = z;
  }
  return NULL;
}

/* ---- Original serial versions ---- */

#define bfca_min(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, tries, smp) \
  bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MINIMIZE, tries, smp)
#define bfca_max(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, tries, smp) \
  bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MAXIMIZE, tries, smp)

double bfca(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, int maxmin, size_t tries, bool smp) {
  double est, lest, best = 1;
  double a_l, a_u, b_l, b_u, l, u;
  size_t i, t, r;
  for (t = 0; t < tries; ++t) {
    r = rand() % (1 << m);
    for (i = 0; i < m; ++i) X[i] = ((r >> i) % 2) ? L[i] : U[i];
    est = 0;
    lest = -1;
    while (est > lest) {
      for (i = 0; i < m; ++i) {
        if (smp) {
          X[i] = L[i];
          l = maxmin*f(X, S_a, C_a, n_a, m);
          X[i] = U[i];
          u = maxmin*f(X, S_a, C_a, n_a, m);
        } else {
          X[i] = L[i];
          a_l = f(X, S_a, C_a, n_a, m);
          b_l = f(X, S_b, C_b, n_b, m);
          X[i] = U[i];
          a_u = f(X, S_a, C_a, n_a, m);
          b_u = f(X, S_b, C_b, n_b, m);
          l = a_l+b_l;
          if (l != 0) l = maxmin*(a_l/l);
          u = a_u+b_u;
          if (u != 0) u = maxmin*(a_u/u);
        }
        lest = est;
        if (l < u) {
          X[i] = L[i];
          est = l;
        } else {
          X[i] = U[i];
          est = u;
        }
      }
    }
    if (best > est) best = est;
  }
  return maxmin*best;
}

void bf(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp) {
  size_t j;
  unsigned long long int k, i;
  double a, b, y;

  *low = 1.0; *up = 0.0;
  k = 1 << m;
  for (i = 0; i < k; ++i) {
    for (j = 0; j < m; ++j) X[j] = ((i >> j) % 2) ? L[j] : U[j];
    a = f(X, S_a, C_a, n_a, m);
    b = f(X, S_b, C_b, n_b, m);
    if (smp) {
      if (*low > a) *low = a;
      if (*up < b) *up = b;
    } else {
      y = a+b;
      if (y != 0) y = a/y;
      if (*low > y) *low = y;
      if (*up < y) *up = y;
    }
  }
}

void bf_minmax(double *X, bool *S_a, bool *S_b, bool* S_c, bool* S_d, double *C_a,
    double *C_b, double *C_c, double *C_d, double *L, double *U, size_t n_a, size_t n_b,
    size_t n_c, size_t n_d, size_t m, double *low, double *up) {
  size_t j;
  unsigned long long int k, i;
  double a, b, c, d, y, z;

  *low = 1.0; *up = 0.0;
  k = 1 << m;
  for (i = 0; i < k; ++i) {
    for (j = 0; j < m; ++j) X[j] = ((i >> j) % 2) ? L[j] : U[j];
    a = f(X, S_a, C_a, n_a, m);
    b = f(X, S_b, C_b, n_b, m);
    c = f(X, S_c, C_c, n_c, m);
    d = f(X, S_d, C_d, n_d, m);
    y = a+d;
    if (y != 0) y = a/y;
    z = b+c;
    if (z != 0) z = b/z;
    if (*low > y) *low = y;
    if (*up < z) *up = z;
  }
}

/* ---- Parallel versions ---- */

static void bf_parallel(bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp, size_t num_threads) {
  unsigned long long k = 1ULL << m;
  if (num_threads > k) num_threads = k;
  if (num_threads <= 1) {
    double X[BFCA_MAX_M];
    bf(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, low, up, smp);
    return;
  }

  pthread_t threads[num_threads];
  bf_task_t tasks[num_threads];

  unsigned long long chunk = k / num_threads;
  unsigned long long remainder = k % num_threads;
  unsigned long long offset = 0;

  for (size_t t = 0; t < num_threads; ++t) {
    unsigned long long this_end = offset + chunk + (t < remainder ? 1 : 0);
    tasks[t] = (bf_task_t){
      .S_a = S_a, .S_b = S_b, .C_a = C_a, .C_b = C_b,
      .L = L, .U = U, .n_a = n_a, .n_b = n_b, .m = m,
      .start = offset, .end = this_end, .smp = smp
    };
    offset = this_end;
    pthread_create(&threads[t], NULL, bf_worker, &tasks[t]);
  }

  *low = 1.0; *up = 0.0;
  for (size_t t = 0; t < num_threads; ++t) {
    pthread_join(threads[t], NULL);
    if (tasks[t].low < *low) *low = tasks[t].low;
    if (tasks[t].up > *up) *up = tasks[t].up;
  }
}

static void bf_minmax_parallel(bool *S_a, bool *S_b, bool *S_c, bool *S_d,
    double *C_a, double *C_b, double *C_c, double *C_d, double *L, double *U,
    size_t n_a, size_t n_b, size_t n_c, size_t n_d, size_t m,
    double *low, double *up, size_t num_threads) {
  unsigned long long k = 1ULL << m;
  if (num_threads > k) num_threads = k;
  if (num_threads <= 1) {
    double X[BFCA_MAX_M];
    bf_minmax(X, S_a, S_b, S_c, S_d, C_a, C_b, C_c, C_d, L, U, n_a, n_b, n_c, n_d, m, low, up);
    return;
  }

  pthread_t threads[num_threads];
  bf_minmax_task_t tasks[num_threads];

  unsigned long long chunk = k / num_threads;
  unsigned long long remainder = k % num_threads;
  unsigned long long offset = 0;

  for (size_t t = 0; t < num_threads; ++t) {
    unsigned long long this_end = offset + chunk + (t < remainder ? 1 : 0);
    tasks[t] = (bf_minmax_task_t){
      .S_a = S_a, .S_b = S_b, .S_c = S_c, .S_d = S_d,
      .C_a = C_a, .C_b = C_b, .C_c = C_c, .C_d = C_d,
      .L = L, .U = U, .n_a = n_a, .n_b = n_b, .n_c = n_c, .n_d = n_d, .m = m,
      .start = offset, .end = this_end
    };
    offset = this_end;
    pthread_create(&threads[t], NULL, bf_minmax_worker, &tasks[t]);
  }

  *low = 1.0; *up = 0.0;
  for (size_t t = 0; t < num_threads; ++t) {
    pthread_join(threads[t], NULL);
    if (tasks[t].low < *low) *low = tasks[t].low;
    if (tasks[t].up > *up) *up = tasks[t].up;
  }
}

/* ---- Adaptive wrappers ---- */

void optimize_credal(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L,
    double *U, size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp,
    size_t num_threads) {
  if (m <= BFCA_THRESHOLD) {
    bf_parallel(S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, low, up, smp, num_threads);
  } else {
    size_t tries = BFCA_TRIES(m);
    *low = bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MINIMIZE, tries, smp);
    if (smp)
      *up = bfca(X, S_b, S_a, C_b, C_a, L, U, n_b, n_a, m, BFCA_MAXIMIZE, tries, smp);
    else
      *up = bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MAXIMIZE, tries, smp);
  }
}

void optimize_credal_minmax(double *X, bool *S_a, bool *S_b, bool *S_c, bool *S_d,
    double *C_a, double *C_b, double *C_c, double *C_d, double *L, double *U,
    size_t n_a, size_t n_b, size_t n_c, size_t n_d, size_t m, double *low, double *up,
    size_t num_threads) {
  if (m <= BFCA_THRESHOLD) {
    bf_minmax_parallel(S_a, S_b, S_c, S_d, C_a, C_b, C_c, C_d, L, U,
        n_a, n_b, n_c, n_d, m, low, up, num_threads);
  } else {
    size_t tries = BFCA_TRIES(m);
    *low = bfca(X, S_a, S_d, C_a, C_d, L, U, n_a, n_d, m, BFCA_MINIMIZE, tries, false);
    *up  = bfca(X, S_b, S_c, C_b, C_c, L, U, n_b, n_c, m, BFCA_MAXIMIZE, tries, false);
  }
}
