#include "cinf.h"
#include "cutils.h"
#include <string.h>

/* ---- Assumption-based reusable Clingo control ---- */

bool can_reuse_control(program_t *P, bool lstable_sat) {
  /* Assumption-based solving is only safe when the program is the same for all total choices.
   * This excludes: lstable (switches to P->stable), smproblog (special handling),
   * neural rules/ADs (require per-choice backend rule additions). */
  if (P->NR_n > 0 || P->NA_n > 0) return false;
  if (P->sem == LSTABLE_SEMANTICS && lstable_sat) return false;
  if (P->sem == SMPROBLOG_SEMANTICS) return false;
  return true;
}

static bool _lookup_literal(clingo_control_t *C, clingo_symbol_t sym, clingo_literal_t *lit) {
  const clingo_symbolic_atoms_t *atoms;
  clingo_symbolic_atom_iterator_t it;
  bool found;
  if (!clingo_control_symbolic_atoms(C, &atoms)) return false;
  if (!clingo_symbolic_atoms_find(atoms, sym, &it)) return false;
  if (!clingo_symbolic_atoms_is_valid(atoms, it, &found)) return false;
  if (!found) { *lit = 0; return true; }
  if (!clingo_symbolic_atoms_literal(atoms, it, lit)) return false;
  return true;
}

bool init_reuse_control(reuse_control_t *rc, program_t *P) {
  memset(rc, 0, sizeof(*rc));

  /* Calculate total assumptions needed: PF + CF + one per AD (selected outcome). */
  size_t n_ad_total = 0;
  for (size_t i = 0; i < P->AD_n; ++i) n_ad_total += P->AD[i].n;
  rc->n_assumptions = P->PF_n + P->CF_n + n_ad_total;

  /* Create and ground control with all atoms as choices. */
  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &rc->C)) return false;
  if (!setup_config(rc->C, "0", false)) goto error;
  if (!clingo_control_add(rc->C, "base", NULL, 0, P->P)) goto error;
  if (P->gr_P[0])
    if (!clingo_control_add(rc->C, "base", NULL, 0, P->gr_P)) goto error;
  if (!add_all_atoms_as_choice(rc->C, P)) goto error;
  if (!clingo_control_ground(rc->C, GROUND_DEFAULT_PARTS, 1, NULL, NULL)) goto error;

  /* Allocate literal arrays. */
  if (P->PF_n > 0) {
    rc->pf_lits = (clingo_literal_t*) malloc(P->PF_n * sizeof(clingo_literal_t));
    if (!rc->pf_lits) goto error;
  }
  if (P->CF_n > 0) {
    rc->cf_lits = (clingo_literal_t*) malloc(P->CF_n * sizeof(clingo_literal_t));
    if (!rc->cf_lits) goto error;
  }
  if (n_ad_total > 0) {
    rc->ad_lits = (clingo_literal_t*) malloc(n_ad_total * sizeof(clingo_literal_t));
    if (!rc->ad_lits) goto error;
  }
  if (P->AD_n > 0) {
    rc->ad_offsets = (size_t*) malloc(P->AD_n * sizeof(size_t));
    if (!rc->ad_offsets) goto error;
  }
  rc->assumptions = (clingo_literal_t*) malloc(rc->n_assumptions * sizeof(clingo_literal_t));
  if (!rc->assumptions) goto error;

  /* Look up literal IDs for all PF atoms. */
  for (size_t i = 0; i < P->PF_n; ++i)
    if (!_lookup_literal(rc->C, P->PF[i].cl_f, &rc->pf_lits[i])) goto error;
  /* Look up literal IDs for all CF atoms. */
  for (size_t i = 0; i < P->CF_n; ++i)
    if (!_lookup_literal(rc->C, P->CF[i].cl_f, &rc->cf_lits[i])) goto error;
  /* Look up literal IDs for all AD outcome atoms. */
  {
    size_t offset = 0;
    for (size_t i = 0; i < P->AD_n; ++i) {
      rc->ad_offsets[i] = offset;
      for (size_t j = 0; j < P->AD[i].n; ++j) {
        if (!_lookup_literal(rc->C, P->AD[i].cl_F[j], &rc->ad_lits[offset + j])) goto error;
      }
      offset += P->AD[i].n;
    }
  }

  return true;
error:
  free_reuse_control(rc);
  return false;
}

void free_reuse_control(reuse_control_t *rc) {
  if (rc->C) { clingo_control_free(rc->C); rc->C = NULL; }
  free(rc->pf_lits); rc->pf_lits = NULL;
  free(rc->cf_lits); rc->cf_lits = NULL;
  free(rc->ad_lits); rc->ad_lits = NULL;
  free(rc->ad_offsets); rc->ad_offsets = NULL;
  free(rc->assumptions); rc->assumptions = NULL;
}

bool build_assumptions(reuse_control_t *rc, program_t *P, total_choice_t *theta) {
  size_t idx = 0;
  /* CF atoms: positive if true in theta, negative if false. */
  for (size_t i = 0; i < P->CF_n; ++i) {
    clingo_literal_t lit = rc->cf_lits[i];
    if (lit == 0) continue;
    rc->assumptions[idx++] = CHOICE_IS_TRUE(theta, i) ? lit : -lit;
  }
  /* PF atoms: positive if true in theta, negative if false. */
  for (size_t i = 0; i < P->PF_n; ++i) {
    clingo_literal_t lit = rc->pf_lits[i];
    if (lit == 0) continue;
    rc->assumptions[idx++] = CHOICE_IS_TRUE(theta, i + P->CF_n) ? lit : -lit;
  }
  /* AD outcomes: assume the selected outcome positive, all others negative. */
  for (size_t i = 0; i < P->AD_n; ++i) {
    size_t offset = rc->ad_offsets[i];
    uint8_t selected = theta->theta_ad[i];
    for (size_t j = 0; j < P->AD[i].n; ++j) {
      clingo_literal_t lit = rc->ad_lits[offset + j];
      if (lit == 0) continue;
      rc->assumptions[idx++] = (j == selected) ? lit : -lit;
    }
  }
  rc->n_assumptions = idx;
  return true;
}

/* ---- End assumption-based reusable Clingo control ---- */

double prob_total_choice_prob(program_t *P, total_choice_t *theta) {
  prob_fact_t *PF = P->PF;
  size_t PF_n = P->PF_n, AD_n = P->AD_n, CF_n = P->CF_n;
  size_t i = 0;
  double p = 1.0;
  bool t;
  for (; i < PF_n; ++i) {
    t = bitvec_GET(&theta->pf, i + CF_n);
    p *= t*PF[i].p + (!t)*(1.0-PF[i].p);
    if (p == 0.0) return 0.0;
  }
  for (i = 0; i < AD_n; ++i) {
    p *= P->AD[i].P[theta->theta_ad[i]];
    if (p == 0.0) return 0.0;
  }
  return p;
}
double prob_total_choice_neural(program_t *P, total_choice_t *theta, size_t offset, bool train) {
  double p = 1.0;
  size_t r = P->CF_n + P->PF_n;
  size_t m = train*P->batch + (!train)*P->m_test;
  for (size_t i = 0; i < P->NR_n; ++i) {
    float *prob = P->NR[i].P + offset*P->NR[i].o;
    size_t stride = P->NR[i].o*m;
    for (size_t j = 0; j < P->NR[i].n; ++j)
      for (size_t o = 0; o < P->NR[i].o; ++o) {
        bool t = bitvec_GET(&theta->pf, r++);
        double q = prob[j*stride+o];
        p *= t*q + (!t)*(1.0-q);
        if (p == 0.0) return 0.0;
      }
  }
  r = P->AD_n;
  for (size_t i = 0; i < P->NA_n; ++i) {
    float *prob = P->NA[i].P + offset*P->NA[i].v*P->NA[i].o;
    size_t vo = P->NA[i].v*P->NA[i].o;
    for (size_t j = 0; j < P->NA[i].n; ++j)
      for (size_t o = 0; o < P->NA[i].o; ++o) {
        p *= prob[j*m*vo + o*P->NA[i].v + theta->theta_ad[r++]];
        if (p == 0.0) return 0.0;
      }
  }
  return p;
}
double prob_total_choice(program_t *P, total_choice_t *theta) {
  return prob_total_choice_prob(P, theta)*prob_total_choice_neural(P, theta, 0, false);
}
double prob_total_choice_ground(array_prob_fact_t *PF, total_choice_t *theta) {
  double p = 1.0;
  bool t;
  for (size_t i = 0; i < PF->n; ++i) {
    t = bitvec_GET(&theta->pf, i);
    p *= t*PF->d[i].p + (!t)*(1.0-PF->d[i].p);
    if (p == 0.0) return 0.0;
  }
  return p;
}

bool init_storage(storage_t *s, program_t *P, array_bool_t (*Pn)[4],
    array_double_t (*K)[4], size_t id, bool *busy_procs, pthread_mutex_t *mu,
    pthread_mutex_t *wakeup, pthread_cond_t *avail, bool lstable_sat, size_t total_choice_n,
    annot_disj_t *ad, size_t ad_n) {
  s->cond_1 = s->cond_2 = s->cond_3 = s->cond_4 = NULL;
  s->count_q_e = s->count_e = s->count_partial_q_e = NULL;
  s->a = s->b = s->c = s->d = NULL;
  s->reuse = NULL;
  s->Pn = Pn; s->K = K; s->P = P;
  s->mu = mu; s->wakeup = wakeup; s->avail = avail;
  if (!setup_conds(&s->cond_1, &s->cond_2, &s->cond_3, &s->cond_4, P->Q_n*sizeof(bool))) goto error;
  if (!setup_counts(&s->count_q_e, &s->count_e, &s->count_partial_q_e, P->Q_n*sizeof(size_t))) goto error;
  if (!setup_map_mappings(P, &s->maps)) goto error;
  if (!P->CF_n) { if (!setup_abcd(&s->a, &s->b, &s->c, &s->d, P->Q_n, sizeof(double))) goto error; }
  s->busy_procs = busy_procs; s->lstable_sat = lstable_sat;
  s->pid = id;
  s->fail = s->warn = false;
  if (!init_total_choice(&s->theta, total_choice_n, P)) goto error;
  return true;
error:
  PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for init_storage!");
  return false;
}

void free_storage_map_mappings(storage_t *s) {
  size_t i_map = 0;
  for (size_t i = 0; i < s->P->Q_n; ++i) {
    if (s->P->Q[i].O_n > 0) free_contents_map_mapping(&s->maps[i_map++]);
  }
  free(s->maps);
}

void free_storage_contents(storage_t *s) {
  free(s->cond_1); free(s->cond_2); free(s->cond_3); free(s->cond_4);
  free(s->count_q_e); free(s->count_e); free(s->count_partial_q_e);
  if (!s->P->CF_n) { free(s->a); free(s->b); free(s->c); free(s->d); }
  free_total_choice_contents(&s->theta);
  free_storage_map_mappings(s);
}

bool setup_conds(bool **cond_1, bool **cond_2, bool **cond_3, bool **cond_4, size_t n) {
  *cond_1 = (bool*) malloc(n);
  if (!(*cond_1)) goto nomem;
  *cond_2 = (bool*) malloc(n);
  if (!(*cond_2)) goto nomem;
  *cond_3 = (bool*) malloc(n);
  if (!(*cond_3)) goto nomem;
  *cond_4 = (bool*) malloc(n);
  if (!(*cond_4)) goto nomem;
  return true;
nomem:
  free(*cond_1); free(*cond_2);
  free(*cond_3); free(*cond_4);
  *cond_1 = *cond_2 = *cond_3 = *cond_4 = NULL;
  return false;
}

bool setup_counts(size_t **count_q_e, size_t **count_e, size_t **count_partial_q_e, size_t n) {
  *count_q_e = (size_t*) malloc(n);
  if (!(*count_q_e)) goto nomem;
  *count_e = (size_t*) malloc(n);
  if (!(*count_e)) goto nomem;
  if (count_partial_q_e) {
    *count_partial_q_e = (size_t*) malloc(n);
    if (!(*count_partial_q_e)) goto nomem;
  }
  return true;
nomem:
  free(*count_q_e); free(*count_e); free(*count_partial_q_e);
  *count_q_e = *count_e = *count_partial_q_e = NULL;
  return false;
}

bool setup_abcd(double **a, double **b, double **c, double **d, size_t n, size_t s) {
  *a = (double*) calloc(n, s);
  if (!(*a)) goto nomem;
  *b = (double*) calloc(n, s);
  if (!(*b)) goto nomem;
  if (c) {
    *c = (double*) calloc(n, s);
    if (!(*c)) goto nomem;
  } if (d) {
    *d = (double*) calloc(n, s);
    if (!(*d)) goto nomem;
  }
  return true;
nomem:
  free(*a); free(*b);
  free(*c); free(*d);
  *a = *b = *c = *d = NULL;
  return false;
}

bool setup_map_mappings(program_t *P, map_mapping_t **maps) {
  size_t n = 0;
  for (size_t i = 0; i < P->Q_n; ++i)
    if (P->Q[i].O_n > 0) ++n;
  if (!n) {
    *maps = NULL;
    return true;
  }
  *maps = (map_mapping_t*) malloc(n*sizeof(map_mapping_t));
  if (!*maps) return false;
  size_t i_map = 0;
  for (size_t i = 0; i < P->Q_n; ++i) {
    if (P->Q[i].O_n <= 0) continue;
    if (!init_map_mapping(&((*maps)[i_map++]), &P->Q[i])) {
      for (size_t j = 0; j < i_map; ++j) free_contents_map_mapping(&((*maps)[j]));
      free(*maps);
      return false;
    }
  }
  return true;
}

bool _init_total_choice(total_choice_t *theta, size_t n, size_t m) {
  if (!bitvec_init(&theta->pf, n)) return false;
  bitvec_zeron(&theta->pf, n);
  theta->ad_n = m;
  theta->theta_ad = (uint8_t*) calloc(m, sizeof(uint8_t));
  return true;
}
bool init_total_choice(total_choice_t *theta, size_t n, program_t *P) {
  size_t l = P->AD_n;
  for (size_t i = 0; i < P->NA_n; ++i) l += P->NA[i].n*P->NA[i].o;
  return _init_total_choice(theta, n, l);
}
void free_total_choice_contents(total_choice_t *theta) {
  bitvec_free_contents(&theta->pf);
  free(theta->theta_ad);
}

size_t get_num_facts(program_t *P) {
  size_t n = P->PF_n + P->CF_n;
  for (size_t i = 0; i < P->NR_n; ++i) n += P->NR[i].n*P->NR[i].o;
  return n;
}

total_choice_t* copy_total_choice(total_choice_t *src, total_choice_t *dst) {
  if (!dst) {
    dst = (total_choice_t*) malloc(sizeof(total_choice_t));
    if (!_init_total_choice(dst, src->pf.n, src->ad_n)) return NULL;
  } else dst->ad_n = src->ad_n;
  bitvec_copy(&src->pf, &dst->pf);
  if (src->ad_n > 0) memcpy(dst->theta_ad, src->theta_ad, src->ad_n*sizeof(uint8_t));
  return dst;
}

bool incr_total_choice(total_choice_t *theta) {
  return !theta->pf.n ? false : bitvec_incr(&theta->pf);
}
bool _incr_total_choice_ad(uint8_t *theta, annot_disj_t *ad, size_t i, size_t ad_n) {
  if (!ad_n) return true;
  if (i == ad_n-1) return (theta[i] = (theta[i] + 1) % ad[i].n) == 0;
  bool c = _incr_total_choice_ad(theta, ad, i+1, ad_n);
  bool l = theta[i] == ad[i].n-1;
  theta[i] = (theta[i] + c) % ad[i].n;
  return c && l;
}
bool _incr_total_choice_nad(uint8_t *theta, neural_annot_disj_t *nad, size_t i, size_t j, size_t a,
    size_t nad_n) {
  if (!nad_n) return true;
  if (a == nad_n-1) return (theta[a] = (theta[a] + 1) % nad[i].v) == 0;
  if (j == nad[i].n*nad[i].o) j = 0, ++i;
  bool c = _incr_total_choice_nad(theta, nad, i, j+1, a+1, nad_n);
  bool l = theta[a] == nad[i].v-1;
  theta[a] = (theta[a] + c) % nad[i].v;
  return c && l;
}
/**
 * Recursive implementation of incrementing total_choice_t ADs.
 */
bool incr_total_choice_ad(total_choice_t *theta, program_t *P) {
  return !(_incr_total_choice_ad(theta->theta_ad, P->AD, 0, P->AD_n) &&
    _incr_total_choice_nad(theta->theta_ad + P->AD_n, P->NA, 0, 0, 0, theta->ad_n - P->AD_n));
}

void print_total_choice(total_choice_t *theta) {
  wprintf(L"Total choice:\nPF: ");
  bitvec_wprint(&theta->pf);
  for (size_t i = 0; i < theta->ad_n; ++i)
    wprintf(L"AD[%lu] = %u\n", i, theta->theta_ad[i]);
}

size_t estimate_nprocs(size_t total_choice_n) {
  /*return (total_choice_n > log2(NUM_PROCS)) ? NUM_PROCS : (1 << total_choice_n);*/
  return NUM_PROCS;
}

int retr_free_proc(bool *busy_procs, size_t num_procs, pthread_mutex_t *wakeup,
    pthread_cond_t *avail) {
  size_t i;
  int id = -1;
  /* The line below does not produce a problematic race condition since it will, at worst, skip
   * the i-th busy_procs and have to iterate NUM_PROCS all over again. */
  pthread_mutex_lock(wakeup);
  while (true) {
    for (i = 0, id = -1; i < num_procs; ++i) {
      if (!busy_procs[i]) { id = i; break; }
    }
    if (id != -1) break;
    pthread_cond_wait(avail, wakeup);
  }
  busy_procs[id] = true;
  pthread_mutex_unlock(wakeup);
  return id;
}

bool dispatch_job_with_payload(total_choice_t *theta, pthread_mutex_t *wakeup, bool *busy_procs,
    storage_t *S, size_t num_procs, threadpool pool, pthread_cond_t *avail, int id,
    void (*compute_func)(void*), void *payload) {
  copy_total_choice(theta, &S[id].theta);
  return !(S[id].fail || thpool_add_work(pool, compute_func, payload));
}
bool dispatch_job(total_choice_t *theta, pthread_mutex_t *wakeup, bool *busy_procs, storage_t *S,
    size_t num_procs, threadpool pool, pthread_cond_t *avail, void (*compute_func)(void*)) {
  int id = retr_free_proc(busy_procs, num_procs, wakeup, avail);
  return dispatch_job_with_payload(theta, wakeup, busy_procs, S, num_procs, pool, avail, id,
      compute_func, (void*) &S[id]);
}

bool add_facts_from_total_choice(clingo_control_t *C, array_prob_fact_t *PF, total_choice_t *theta) {
  clingo_backend_t *back;
  if (!clingo_control_backend(C, &back)) return false;
  if (!clingo_backend_begin(back)) goto cleanup;
  for (size_t i = 0; i < PF->n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i)) continue;
    if (!clingo_backend_add_atom(back, &PF->d[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
cleanup:
  if (!clingo_backend_end(back)) return false;
  return true;
}

void join_ad_choice(char *dst, char **src, size_t n) {
  *dst++ = '{';
  for (size_t i = 0; i < n; ++i, ++dst) {
    for (char *t = src[i]; *t; ++t) *dst++ = *t;
    *dst = ';';
  }
  *(dst-1) = '}'; dst[0] = '='; dst[1] = '1'; dst[2] = '.'; dst[3] = '\0';
}

bool join_ad_choice_sym(char *dst, size_t max_size, clingo_symbol_t *S, size_t n) {
  *dst++ = '{';
  for (size_t i = 0; i < n; ++i, ++dst) {
    if (!clingo_symbol_to_string(S[i], dst, max_size)) return false;
    dst += strlen(dst); *dst = ';';
  }
  *(dst-1) = '}'; dst[0] = '='; dst[1] = '1'; dst[2] = '.'; dst[3] = '\0';
  return true;
}

bool add_all_atoms_as_choice(clingo_control_t *C, program_t *P) {
  bool ok = false;
  clingo_backend_t *back;
  clingo_atom_t *heads = NULL;
  size_t nheads = P->PF_n + P->CF_n + P->NR_n;

  /* Get the control's backend. */
  if (!clingo_control_backend(C, &back)) return false;
  /* Startup the backend. */
  if (!clingo_backend_begin(back)) goto cleanup;

  /* Collect all probabilistic facts. */
  heads = (clingo_atom_t*) malloc(nheads*sizeof(clingo_atom_t));
  if (!heads) goto cleanup;
  size_t i_head = 0;
  for (size_t i = 0; i < P->PF_n; ++i)
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &heads[i_head++])) goto cleanup;
  for (size_t i = 0; i < P->CF_n; ++i)
    if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &heads[i_head++])) goto cleanup;
  for (size_t i = 0; i < P->NR_n; ++i)
    for (size_t j = 0; j < P->NR[i].n; ++j)
      if (!clingo_backend_add_atom(back, &P->NR[i].H[j], &heads[i_head++])) goto cleanup;

  /* Add them to the backend as a single choice. */
  if (!clingo_backend_rule(back, true, heads, nheads, NULL, 0)) goto cleanup;

  /* We're done with backend. */
  if (!clingo_backend_end(back)) { back = NULL; goto cleanup; }
  back = NULL;

  /* Add each AD as a choice with cardinality constraint of one. */
  char ad_choices[2048];
  for (size_t i = 0; i < P->AD_n; ++i) {
    join_ad_choice(ad_choices, (char**) P->AD[i].F, P->AD[i].n);
    if (!clingo_control_add(C, "base", NULL, 0, ad_choices)) goto cleanup;
  }
  /* Add neural ADs the same way. */
  for (size_t i = 0; i < P->NA_n; ++i)
    for (size_t j = 0; j < P->NA[i].n; ++j) {
      if (!join_ad_choice_sym(ad_choices, 2048, P->NA[i].H+j*P->NA[i].v, P->NA[i].v)) goto cleanup;
      if (!clingo_control_add(C, "base", NULL, 0, ad_choices)) goto cleanup;
    }
  ok = true;
cleanup:
  free(heads);
  /* Cleanup backend. */
  if (back) if (!clingo_backend_end(back)) return false;
  return ok;
}

bool add_neural_rule_atoms(clingo_backend_t *back, program_t *P, total_choice_t *theta) {
  clingo_atom_t h;
  clingo_literal_t B[64];
  size_t c = P->CF_n + P->PF_n;
  for (size_t i = 0; i < P->NR_n; ++i)
    for (size_t j = 0; j < P->NR[i].n; ++j) {
      /* Record body literals. */
      for (size_t b = 0; b < P->NR[i].k; ++b) {
        size_t u = j*P->NR[i].k+b;
        if (!clingo_backend_add_atom(back, &P->NR[i].B[u], (clingo_atom_t*) &B[b]))
          return false;
        if (!P->NR[i].S[u]) B[b] = -B[b];
      }
      /* Select head from outcomes. */
      for (size_t o = 0; o < P->NR[i].o; ++o) {
        if (!CHOICE_IS_TRUE(theta, c++)) continue;
        if (!clingo_backend_add_atom(back, &P->NR[i].H[j*P->NR[i].o+o], &h)) return false;
        /* Add neural rule. */
        if (!clingo_backend_rule(back, false, &h, 1, B, P->NR[i].k)) return false;
      }
    }
  return true;
}

bool add_neural_ad_atoms(clingo_backend_t *back, program_t *P, total_choice_t *theta) {
  clingo_atom_t h;
  clingo_literal_t B[64];
  size_t r = P->AD_n;
  for (size_t i = 0; i < P->NA_n; ++i)
    for (size_t j = 0; j < P->NA[i].n; ++j) {
      for (size_t b = 0; b < P->NA[i].k; ++b) {
        size_t u = j*P->NA[i].k+b;
        if (!clingo_backend_add_atom(back, &P->NA[i].B[u], (clingo_atom_t*) &B[b]))
          return false;
        if (!P->NA[i].S[u]) B[b] = -B[b];
      }
      for (size_t o = 0; o < P->NA[i].o; ++o) {
        if (!clingo_backend_add_atom(back, &P->NA[i].H[j*P->NA[i].v*P->NA[i].o +
              o*P->NA[i].v + theta->theta_ad[r++]], &h))
          return false;
        if (!clingo_backend_rule(back, false, &h, 1, B, P->NA[i].k)) return false;
      }
    }
  return true;
}

bool add_atoms_from_total_choice(clingo_control_t *C, program_t *P, total_choice_t *theta) {
  bool ok = false;
  clingo_backend_t *back;
  /* Get the control's backend. */
  if (!clingo_control_backend(C, &back)) return false;
  /* Startup the backend. */
  if (!clingo_backend_begin(back)) goto cleanup;
  /* Add the credal facts according to the total rule. */
  for (size_t i = 0; i < P->CF_n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i)) continue;
    if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the probabilistic facts according to the total rule. */
  for (size_t i = 0; i < P->PF_n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i + P->CF_n)) continue;
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the neural rules according to the total rule. */
  if (!add_neural_rule_atoms(back, P, theta)) goto cleanup;
  /* Add the annotated disjunction rules according to the total rule encoded by theta_ad. */
  for (size_t i = 0; i < P->AD_n; ++i) {
    clingo_atom_t a;
    if (!clingo_backend_add_atom(back, &P->AD[i].cl_F[theta->theta_ad[i]], &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the neural annotated disjunction rules according to the total rule encoded by theta_ad. */
  if (!add_neural_ad_atoms(back, P, theta)) goto cleanup;
  ok = true;
cleanup:
  /* Cleanup backend. */
  if (!clingo_backend_end(back)) return false;
  return ok;
}

bool _prepare_control(clingo_control_t **C, program_t *P, total_choice_t *theta,
    const char *nmodels, bool parallelize_clingo, const char *append) {
  /* Create new clingo controller. */
  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, C)) return false;
  /* Config to enumerate all models. */
  if (!setup_config(*C, nmodels, false)) return false;
  /* Add the purely logical part. */
  if (!clingo_control_add(*C, "base", NULL, 0, P->P)) return false;
  if (append) if (!clingo_control_add(*C, "base", NULL, 0, append)) return false;
  /* Add grounded probabilistic rules. */
  if (P->gr_P[0]) if (!clingo_control_add(*C, "base", NULL, 0, P->gr_P)) return false;
  if (theta) if (!add_atoms_from_total_choice(*C, P, theta)) return false;
  return true;
}

bool prepare_control_preground(clingo_control_t **C, program_t *P, total_choice_t *theta,
    const char *nmodels, bool parallelize_clingo, const char *append, array_prob_fact_t *gr_PF,
    total_choice_t *gr_theta) {
  if (!_prepare_control(C, P, theta, nmodels, parallelize_clingo, append)) return false;
  if (!add_facts_from_total_choice(*C, gr_PF, gr_theta)) return false;
  /* Ground atoms. */
  if (!atomic_ground(*C, NULL, NULL)) return false;
  return true;
}

bool atomic_ground(clingo_control_t *C, clingo_ground_callback_t gcb, void *gdata) {
  /* No mutex needed: each thread operates on its own clingo_control_t instance.
   * prepare_control() already calls clingo_control_ground() without a mutex. */
  return clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, gcb, gdata);
}

bool prepare_control(clingo_control_t **C, program_t *P, total_choice_t *theta,
    const char *nmodels, bool parallelize_clingo, const char *append) {
  if (!_prepare_control(C, P, theta, nmodels, parallelize_clingo, append)) return false;
  if (!clingo_control_ground(*C, GROUND_DEFAULT_PARTS, 1, NULL, NULL)) return false;
  return true;
}

bool setup_config(clingo_control_t *C, const char *nmodels, bool parallelize_clingo) {
  clingo_configuration_t *cfg = NULL;
  clingo_id_t cfg_root, cfg_sub;

  /* Get the control's configuration. */
  if (!clingo_control_configuration(C, &cfg)) return false;
  /* Set to enumerate all stable models. */
  if (!clingo_configuration_root(cfg, &cfg_root)) return false;
  if (!clingo_configuration_map_at(cfg, cfg_root, "solve.models", &cfg_sub)) return false;
  if (!clingo_configuration_value_set(cfg, cfg_sub, nmodels)) return false;
  if (parallelize_clingo) {
    /* Set parallel_mode to "NUM_PROCS,compete", where NUM_PROCS is the #procs in this machine. */
    if (!clingo_configuration_map_at(cfg, cfg_root, "solve.parallel_mode", &cfg_sub)) return false;
    if (!clingo_configuration_value_set(cfg, cfg_sub, NUM_PROCS_CONFIG_STR)) return false;
  }

  return true;
}

bool has_total_model(program_t *P, total_choice_t *theta, bool *has) {
  clingo_control_t *C = NULL;
  clingo_solve_handle_t *handle;
  clingo_solve_result_bitset_t res;
  /* Prepare control according to the stable semantics. */
  if (!prepare_control(&C, P->stable, theta, "1", false, NULL)) goto cleanup;
  /* Solve and determine if there exists a (total) model. */
  if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle)) goto cleanup;
  if (!clingo_solve_handle_get(handle, &res)) goto cleanup;
  *has = (res & clingo_solve_result_satisfiable);
  /* Cleanup. */
  clingo_control_free(C);
  return true;
cleanup:
  clingo_control_free(C);
  return false;
}

size_t num_prob_params(program_t *P) {
  size_t n = P->PF_n + P->NR_n;
  for (size_t i = 0; i < P->AD_n; ++i) n += P->AD[i].n;
  for (size_t i = 0; i < P->NA_n; ++i) n += P->NA[i].v*P->NA[i].n*P->NA[i].o;
  return n;
}

bool neg_partial_cmp(bool x, bool _x, char s) {
  /* See page 36 of the lparse manual. This is the negation of the truth value of an atom. */
  if (s == QUERY_TERM_POS)
    return !(x && _x);
  else if (s == QUERY_TERM_UND)
    return x || !_x; /* ≡ !(!x && _x) */
  /* else s == QUERY_TERM_NEG */
  return _x; /* (x && _x) || (!x && _x) ≡ !(x && _x) && !(!x && _x); */
}

bool model_contains(const clingo_model_t *M, query_t *q, size_t i, bool *c, bool query_or_evi, bool is_partial) {
  clingo_symbol_t x, x_u;
  uint8_t s;
  bool c_x;

  if (query_or_evi) {
    /* Query. */
    x = q->Q[i]; s = q->Q_s[i];
    if (is_partial) x_u = q->Q_u[i];
  } else {
    /* Evidence. */
    x = q->E[i]; s = q->E_s[i];
    if (is_partial) x_u = q->E_u[i];
  }

  if (!clingo_model_contains(M, x, &c_x)) return false;
  if (is_partial) {
    bool c_a;
    if (!clingo_model_contains(M, x_u, &c_a)) return false;
    if (neg_partial_cmp(c_x, c_a, s)) { *c = false; return true; }
  } else {
    if (c_x != s) { *c = false; return true; }
  }
  *c = true;
  return true;
}

