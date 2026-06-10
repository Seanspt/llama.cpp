#pragma once

#include "llama.h"

#include "common.h"

#include <deque>
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

// Compute an ambiguity margin from logits as a lightweight proxy for entropy.
// Returns P(top1) / P(top2) in softmax space — margin close to 1.0 means high
// ambiguity (the model is undecided between top-2 tokens); larger values mean
// higher confidence (deterministic position).
float compute_ambiguity_margin(const float * logits, int n_vocab);

// Check whether a token is sensitive to loose verification: EOS, BOS, control
// tokens, and structured chat markers must always be exact-matched.
bool is_control_sensitive(llama_token tok, const struct llama_vocab * vocab);

// Streaming output buffer that delays output of provisionally-accepted tokens
// until they have cleared the deferred window (W tokens later).
struct fly_output_buffer {
    std::deque<llama_token> pending;
    int window_size = 6;

    void push(llama_token tok) { pending.push_back(tok); }

    void push_batch(const std::vector<llama_token> & toks) {
        for (auto t : toks) pending.push_back(t);
    }

    // Return tokens that are safe to output (outside the W-token danger zone).
    std::vector<llama_token> flushable() {
        std::vector<llama_token> safe;
        while ((int) pending.size() > window_size) {
            safe.push_back(pending.front());
            pending.pop_front();
        }
        return safe;
    }

    // Roll back to only keep the first n tokens (called on rejection).
    void reject(int n_keep) {
        if (n_keep < (int) pending.size()) {
            pending.resize(n_keep);
        }
    }

    // Flush all remaining tokens (e.g. on EOS or generation end).
    std::vector<llama_token> flush_all() {
        std::vector<llama_token> all;
        while (!pending.empty()) {
            all.push_back(pending.front());
            pending.pop_front();
        }
        return all;
    }

    void clear() { pending.clear(); }
    size_t size() const { return pending.size(); }
};

// FLy loose verification: replaces the exact-match acceptance loop with a
// two-tier mechanism (ambiguity gate + deferred window).
//
// Returns at least 1 token (the bonus), up to draft.size() + 1 tokens.
// The semantics are identical to common_sampler_sample_and_accept_n().
std::vector<llama_token> common_sampler_sample_and_accept_n_fly(
    struct common_sampler * gsmpl,
    struct llama_context * ctx,
    const std::vector<int> & idxs,
    const llama_tokens & draft,
    const common_params_speculative_fly & params,
    bool grammar_first = false);
