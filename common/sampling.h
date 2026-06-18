#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;

//
// FLy (Training-Free Loosely Speculative Decoding)
//

struct common_params_speculative_fly;

// Per-step FLy verification statistics.
// Callers accumulate across steps to get aggregate counts.
struct common_fly_stats {
    // ── Acceptance / rejection counts ──
    int n_loose_accept   = 0;  // mismatch accepted via deferred window
    int n_strict_reject   = 0;  // margin >= threshold (margin gate) or ΔlogP >= τ
    int n_control_reject  = 0;  // special-token hard block
    int n_window_reject   = 0;  // window contained another mismatch
    int n_boundary_reject = 0;  // not enough lookahead (K > W)

    // ── Named kill paths (subset of n_strict_reject) ──
    int n_margin_kill = 0;      // rejected by margin gate (MARGIN-KILL-SHORT)
    int n_delta_kill  = 0;      // rejected by ΔlogP gate (DLP-KILL-*)

    // ── P1 (analytical pass) ──
    int n_total_draft = 0;      // total draft positions inspected
    int n_total_match = 0;      // exact match positions
    int n_total_miss  = 0;      // mismatch positions (= n_total_draft - n_total_match after final merge)

    // ── ΔlogP distribution (loose accepts only) ──
    // ΔlogP = log P(target_top1) - log P(draft_token)
    // Positive value means the target model preferred its own top-1 over the
    // draft token. Loose-accepted tokens carry non-zero ΔlogP — this is the
    // probability mass "given up" for acceleration.
    double sum_delta_logp  = 0.0;
    int    n_delta_logp    = 0;
    float  delta_logp_min  =  INFINITY;
    float  delta_logp_max  = -INFINITY;
    int    n_delta_zero    = 0;   // loose accepts where delta_logp == 0 (gate blind spot)
    int    n_delta_ge_tau  = 0;   // loose accepts where delta_logp >= τ (should be 0)

    // ── Margin distribution (loose accepts only) ──
    float  margin_min      =  INFINITY;
    float  margin_max      = -INFINITY;
    double sum_margin      = 0.0;  // for computing mean

    // ── Helpers ──

    float avg_delta_logp() const {
        return n_delta_logp > 0 ? (float)(sum_delta_logp / n_delta_logp) : 0.0f;
    }

    float avg_margin() const {
        return n_delta_logp > 0 ? (float)(sum_margin / n_delta_logp) : 0.0f;
    }

    void merge(const common_fly_stats & other) {
        n_loose_accept   += other.n_loose_accept;
        n_strict_reject   += other.n_strict_reject;
        n_control_reject  += other.n_control_reject;
        n_window_reject   += other.n_window_reject;
        n_boundary_reject += other.n_boundary_reject;
        n_margin_kill     += other.n_margin_kill;
        n_delta_kill      += other.n_delta_kill;
        n_total_draft     += other.n_total_draft;
        n_total_match     += other.n_total_match;
        n_total_miss      += other.n_total_miss;
        sum_delta_logp    += other.sum_delta_logp;
        n_delta_logp      += other.n_delta_logp;
        delta_logp_min     = std::min(delta_logp_min, other.delta_logp_min);
        delta_logp_max     = std::max(delta_logp_max, other.delta_logp_max);
        n_delta_zero      += other.n_delta_zero;
        n_delta_ge_tau    += other.n_delta_ge_tau;
        margin_min         = std::min(margin_min, other.margin_min);
        margin_max         = std::max(margin_max, other.margin_max);
        sum_margin        += other.sum_margin;
    }

    void reset() {
        n_loose_accept   = 0;
        n_strict_reject   = 0;
        n_control_reject  = 0;
        n_window_reject   = 0;
        n_boundary_reject = 0;
        n_margin_kill     = 0;
        n_delta_kill      = 0;
        n_total_draft     = 0;
        n_total_match     = 0;
        n_total_miss      = 0;
        sum_delta_logp    = 0.0;
        n_delta_logp      = 0;
        delta_logp_min    =  INFINITY;
        delta_logp_max    = -INFINITY;
        n_delta_zero      = 0;
        n_delta_ge_tau    = 0;
        margin_min        =  INFINITY;
        margin_max        = -INFINITY;
        sum_margin        = 0.0;
    }
};

// Lightweight proxy for the paper's normalized entropy h_j (see Algorithm 1).
//
// Returns P(top1) / P(top2), which equals exp(logit_max1 - logit_max2).
//   ≈ 1.0  → top-1 and top-2 nearly tied (high ambiguity → defer)
//   ≫ 1.0  → top-1 dominates (low ambiguity      → strict reject)
//
// Only requires one O(|V|) scan for the two largest logits, avoiding a full
// softmax + entropy computation. See the implementation comment for the
// calibration of ambiguity_threshold against the paper's θ.
float compute_ambiguity_margin(const float * logits, int n_vocab);

// Check whether a token is sensitive to loose verification: EOS, BOS, control
// tokens, and structured chat markers must always be exact-matched.
bool is_control_sensitive(llama_token tok, const struct llama_vocab * vocab);

// FLy loose verification: replaces the exact-match acceptance loop with a
// two-tier mechanism (ambiguity gate + deferred window).
//
// Returns at least 1 token (the bonus), up to draft.size() + 1 tokens.
// The semantics are identical to common_sampler_sample_and_accept_n().
//
// If stochastic is true (T>0), a cloned sampler is used in the analytical
// phase to extract target tokens via the full sampling chain. If false (T=0),
// argmax is used directly on raw logits — faster and deterministic.
std::vector<llama_token> common_sampler_sample_and_accept_n_fly(
    struct common_sampler * gsmpl,
    struct llama_context * ctx,
    const std::vector<int> & idxs,
    const llama_tokens & draft,
    const common_params_speculative_fly & params,
    bool grammar_first = false,
    bool stochastic      = false,
    common_fly_stats * stats = nullptr);
