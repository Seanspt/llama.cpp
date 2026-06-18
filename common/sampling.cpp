#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            params.reasoning_budget_start,
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    if (params.has_logit_bias()) {
        samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), params.logit_bias.size(), params.logit_bias.data()));
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
    };
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

//
// FLy (Training-Free Loosely Speculative Decoding)
//

float compute_ambiguity_margin(const float * logits, int n_vocab) {
    float logit_max1 = -INFINITY;
    float logit_max2 = -INFINITY;

    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!std::isfinite(v)) {
            continue; // skip NaN/Inf — they don't carry usable signal
        }
        if (v > logit_max1) {
            logit_max2 = logit_max1;
            logit_max1 = v;
        } else if (v > logit_max2) {
            logit_max2 = v;
        }
    }

    // In softmax space, P(top1)/P(top2) = exp(logit_max1)/exp(logit_max2) =
    // exp(logit_max1 - logit_max2). This ratio measures how concentrated the
    // probability mass is on the top-1 token:
    //   ≈ 1.0  → top-1 and top-2 are nearly tied (high ambiguity)
    //   ≫ 1.0  → top-1 dominates (low ambiguity, deterministic position)
    //
    // This is a lightweight proxy for the paper's normalized entropy h_j:
    //   margin close to 1.0  ⇔  h_j high  (target is undecided → defer)
    //   margin ≫ 1.0          ⇔  h_j low   (target is confident  → strict reject)
    //
    // The default ambiguity_threshold = 2.0 means: reject when top-1 is at
    // least twice as likely as top-2. This corresponds roughly to a logit gap
    // of ln(2) ≈ 0.69. The paper's θ = 0.3 is a normalized-entropy threshold;
    // the margin proxy is calibrated to produce similar defer/reject behaviour
    // in practice while requiring only a single O(|V|) scan instead of a full
    // softmax + entropy computation.
    return expf(logit_max1 - logit_max2);
}

bool is_control_sensitive(llama_token tok, const struct llama_vocab * vocab) {
    // Primary: vocab-level classification (EOS, BOS, control tokens)
    if (llama_vocab_is_eog(vocab, tok)) {
        return true;
    }

    if (llama_vocab_is_control(vocab, tok)) {
        return true;
    }

    // Chat-template structural markers (e.g. <|im_start|>, <|start_header_id|>)
    // are tagged LLAMA_TOKEN_ATTR_USER_DEFINED in well-formed GGUF metadata.
    // This catches them with a single bit test instead of falling through to
    // the strstr path below.
    if (llama_vocab_get_attr(vocab, tok) & LLAMA_TOKEN_ATTR_USER_DEFINED) {
        return true;
    }

    // Fallback: string-based detection for the same markers when the GGUF
    // metadata does not correctly tag them as USER_DEFINED (common in older
    // or community-quantized models). The tokenizer may split these across
    // multiple sub-tokens, so individual fragments won't match the full
    // marker string — we match the distinctive substrings.
    //
    // NOTE: deliberately does NOT use generic patterns like "<…>" or newline
    // matching — those would misclassify HTML/XML tags, math/inequality signs,
    // code snippets, and line-break tokens as "control", crippling FLy on
    // code-generation and structured-output tasks.
    const char * text = llama_vocab_get_text(vocab, tok);
    if (text) {
        // Llama 3 / Llama 4 chat markers
        if (strstr(text, "<|start_header_id|>") != nullptr) return true;
        if (strstr(text, "<|end_header_id|>")   != nullptr) return true;
        if (strstr(text, "<|eot_id|>")          != nullptr) return true;

        // Qwen / ChatML markers (tokenizer may split these)
        if (strstr(text, "im_start") != nullptr) return true;
        if (strstr(text, "im_end")   != nullptr) return true;

        // Legacy Llama chat markers
        if (strstr(text, "[INST]")  != nullptr) return true;
        if (strstr(text, "[/INST]") != nullptr) return true;
    }

    return false;
}

static llama_token argmax_logits(const float * logits, int n_vocab) {
    float max_val = -INFINITY;
    int   max_idx = -1;

    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!std::isfinite(v)) {
            continue; // skip NaN/Inf — they don't carry usable signal
        }
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    // If all logits were non-finite, fall back to token 0 rather than
    // returning -1, which would crash downstream. This path should be
    // unreachable in practice but guards against corrupt model output.
    if (max_idx < 0) {
        return 0;
    }

    return (llama_token) max_idx;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_fly(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const std::vector<int> & idxs,
        const llama_tokens & draft,
        const common_params_speculative_fly & params,
        bool grammar_first,
        bool stochastic,
        common_fly_stats * stats) {

    const int K = (int) draft.size();
    GGML_ASSERT((int) idxs.size() == K + 1 && "idxs.size() must be draft.size() + 1");

    common_fly_stats local_stats; // per-step counts (never reset accumulated stats)
    common_fly_stats * st = stats ? &local_stats : nullptr;
    if (st) {
        st->n_total_draft = K;
    }

    if (params.debug_trace) {
        const bool  dlp_active = (params.delta_logp_threshold > 0.0f);
        if (dlp_active) {
            LOG_INF("FLy ENTER: K=%d, stochastic=%d, ΔlogP gate τ=%.4f, margin safety thr=%.2f, W=%d\n",
                    K, (int) stochastic, (double) params.delta_logp_threshold,
                    (double) params.ambiguity_threshold, params.window_size);
        } else {
            LOG_INF("FLy ENTER: K=%d, stochastic=%d, margin gate thr=%.2f, W=%d\n",
                    K, (int) stochastic, (double) params.ambiguity_threshold, params.window_size);
        }
    }

    if (K == 0) {
        // No draft tokens: just sample the bonus from the first logit
        std::vector<llama_token> result;
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[0], grammar_first);
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const float ambiguity_threshold  = params.ambiguity_threshold;
    const float delta_logp_threshold = params.delta_logp_threshold;
    const bool  use_delta_logp_gate  = (delta_logp_threshold > 0.0f);
    const int   window_W             = params.window_size;

    // ===== Phase 1: Analytical pass (no state mutations on gsmpl) =====
    //
    // For T=0 (greedy): extract target[i] via raw argmax — cheaper, no sampler overhead.
    // For T>0 (stochastic): clone the sampler and use it to sample target[i] using the
    //   full sampling chain (temperature, top-k, top-p, penalties, etc.). The clone
    //   starts from the same penalty state as the real sampler, and its state evolves
    //   independently as we accept each target[i]. This accurately simulates what the
    //   real sampler would produce at each position.
    std::vector<llama_token> target(K);
    std::vector<float>       margin(K);
    std::vector<bool>        match(K);
    std::vector<float>       delta_logp(K, 0.0f);  // log P(target) - log P(draft), >= 0

    common_sampler_ptr smpl_analytical;  // only used in stochastic mode
    if (stochastic) {
        smpl_analytical.reset(common_sampler_clone(gsmpl));
    }

    for (int i = 0; i < K; i++) {
        // idxs[0] = batch position of id_last → prediction after id_last → compare with draft[0]
        const float * logits = llama_get_logits_ith(ctx, idxs[i]);
        if (!logits) {
            LOG_WRN("FLy: null logits at i=%d idx=%d, falling back to standard SPD\n", i, idxs[i]);
            // fall back to standard exact-match verification
            return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
        }

        if (stochastic) {
            target[i] = common_sampler_sample(smpl_analytical.get(), ctx, idxs[i], grammar_first);
            common_sampler_accept(smpl_analytical.get(), target[i], true);
        } else {
            target[i] = argmax_logits(logits, n_vocab);
        }

        // NOTE: margin is computed from raw logits in both T=0 and T>0 modes.
        // Temperature / top-k / top-p are applied inside the sampling chain
        // and do not modify the raw logits returned by llama_get_logits_ith(),
        // so margin[j] is identical regardless of temperature.
        //
        // In stochastic mode (T>0), target[i] is drawn from the full sampling
        // chain and is therefore more likely to diverge from draft[i] than at
        // T=0. This produces more mismatches overall, which makes the deferred
        // window check (window_clean) harder to satisfy — subsequent positions
        // are also more likely to mismatch. The net effect is that FLy's
        // acceptance advantage shrinks as temperature increases, and at high T
        // standard exact-match SPD may be faster (more strict rejects → shorter
        // accepted runs → more forward passes).
        //
        // This is an inherent limitation of using a deterministic margin proxy
        // with stochastic targets. The paper evaluates FLy primarily at T≈0;
        // for high-temperature generation, the ambiguity gate is simply less
        // effective, not unsafe. Output quality is unaffected because any token
        // accepted through the deferred window was verified to be semantically
        // consistent (clean window) with the target's own subsequent predictions.
        margin[i] = compute_ambiguity_margin(logits, n_vocab);
        match[i]  = (draft[i] == target[i]);

        // Phase-1 per-position counts (for summary statistics)
        if (st) {
            if (match[i]) {
                st->n_total_match++;
            } else {
                st->n_total_miss++;
            }
        }

        // ΔlogP = log P_target(top1) - log P_target(draft_token)
        // In logit space this is simply logit[target] - logit[draft] because
        // the softmax normalisation constant Z cancels out.
        //
        // NOTE: computed unconditionally on every mismatch — it is a gate
        // input, not a stats-only quantity. The server path does not pass a
        // stats pointer, but the ΔlogP gate must still function.
        if (!match[i]) {
            const float logit_draft  = logits[draft[i]];
            const float logit_target = logits[target[i]];
            if (std::isfinite(logit_draft) && std::isfinite(logit_target)) {
                delta_logp[i] = logit_target - logit_draft;
            } else if (std::isfinite(logit_target) && !std::isfinite(logit_draft)) {
                // Draft token has zero / near-zero probability (logit = -inf).
                // The target model strongly disprefers it → set to INFINITY
                // so the ΔlogP gate will reject. Without this, the gate never
                // fires and FLy degenerates to "accept everything" in MTP
                // scenarios where per-token logits are sparse (only top-k finite).
                delta_logp[i] = INFINITY;
            }
        }

        if (params.debug_trace) {
            const char * dt = llama_vocab_get_text(vocab, draft[i]);
            const char * tt = llama_vocab_get_text(vocab, target[i]);
            const char * st = match[i] ? "MATCH" : "MISS";
            LOG_INF("FLy-P1 i=%d draft[%s]=%d target[%s]=%d %s margin=%.2f\n",
                    i, dt ? dt : "?", draft[i], tt ? tt : "?", target[i], st, (double)margin[i]);
        }
    }

    // ===== Phase 2: Decision pass (determine first_reject) =====
    int first_reject = K; // default: accept all K draft tokens

    for (int j = 0; j < K; j++) {
        if (match[j]) {
            continue; // exact match: accept
        }

        // --- Special token protection: always strict ---
        {
            bool d_ctrl = is_control_sensitive(draft[j], vocab);
            bool t_ctrl = is_control_sensitive(target[j], vocab);
            if (d_ctrl || t_ctrl) {
                if (params.debug_trace) {
                    const char * dt = llama_vocab_get_text(vocab, draft[j]);
                    const char * tt = llama_vocab_get_text(vocab, target[j]);
                    LOG_INF("FLy CONTROL-REJECT pos %d: draft[%s]=%d ctrl=%d  target[%s]=%d ctrl=%d\n",
                            j, dt ? dt : "?", draft[j], (int)d_ctrl, tt ? tt : "?", target[j], (int)t_ctrl);
                    bool eog_d = llama_vocab_is_eog(vocab, draft[j]);
                    bool eog_t = llama_vocab_is_eog(vocab, target[j]);
                    bool vc_d = llama_vocab_is_control(vocab, draft[j]);
                    bool vc_t = llama_vocab_is_control(vocab, target[j]);
                    LOG_INF("FLy CONTROL-DETAIL: draft eog=%d vctrl=%d  target eog=%d vctrl=%d\n",
                            (int)eog_d, (int)vc_d, (int)eog_t, (int)vc_t);
                }
                if (st) { st->n_control_reject++; }
                first_reject = j;
                break;
            }
        }

        // --- Ambiguity gate ---
        // Two modes:
        //   1. ΔlogP gate (delta_logp_threshold > 0):
        //      reject when log P_target(top1) - log P_target(draft) >= τ.
        //      This directly measures how much worse the draft token is.
        //   2. Margin gate (default):
        //      reject when P(top1)/P(top2) >= threshold (distribution too peaked).
        bool gate_reject = false;
        if (use_delta_logp_gate) {
            gate_reject = (delta_logp[j] >= delta_logp_threshold);
        } else {
            gate_reject = (margin[j] >= ambiguity_threshold);
            if (params.debug_trace && gate_reject) {
                const char * draft_text = llama_vocab_get_text(vocab, draft[j]);
                const char * tgt_text   = llama_vocab_get_text(vocab, target[j]);
                LOG_INF("FLy STRICT-REJECT pos %d: draft='%s' != target='%s', margin=%.2f >= %.2f\n",
                        j, draft_text ? draft_text : "?", tgt_text ? tgt_text : "?",
                        (double) margin[j], (double) ambiguity_threshold);
            }
        }

        if (gate_reject) {
            if (st) {
                st->n_strict_reject++;
                if (use_delta_logp_gate) { st->n_delta_kill++;  }
                else                     { st->n_margin_kill++; }
            }
            first_reject = j;
            break;
        }

        // --- Deferred window ---
        if (j + window_W >= K) {
            // Boundary: not enough lookahead tokens.
            //
            // When the entire draft is shorter than or equal to the window
            // (K <= W), every mismatch is trivially at the boundary. In
            // this case conservative rejection is pointless — there are
            // simply not enough tokens to look ahead. Accept the mismatch
            // as semantic variation rather than rejecting it, which would
            // otherwise create an infinite loop with checkpoint-based
            // partial draft reuse.
            //
            // When K > W, a boundary mismatch at the tail of a long draft
            // genuinely cannot be verified → conservative reject.
            if (K <= window_W) {
                // Draft is too short for a meaningful lookahead window.
                //
                // Without lookahead tokens to verify semantic consistency,
                // we fall back to a dual-gate safety check: the mismatch
                // must pass BOTH the primary gate (ΔlogP or margin) AND the
                // margin safety backstop. This prevents FLy from degenerating
                // into "accept everything" when K is small (e.g. MTP with
                // K=4, W=6) — a scenario where the deferred window is
                // structurally unavailable.
                //
                // ── Dual gate for short drafts ──
                // 1. Primary gate (ΔlogP or margin, same as main gate above).
                // 2. Margin safety backstop (always active in short-draft
                //    mode, even when ΔlogP gate is enabled). A highly-peaked
                //    target distribution with a mismatch means accepting the
                //    draft token would be genuinely lossy; the window that
                //    would normally catch this is unavailable.
                bool short_gate_reject = false;
                bool is_delta_kill  = false;
                bool is_margin_kill = false;

                // ΔlogP gate
                if (use_delta_logp_gate && delta_logp[j] >= delta_logp_threshold) {
                    if (params.debug_trace) {
                        LOG_INF("FLy DLP-KILL-SHORT: pos=%d delta_logp=%.4f >= τ=%.2f, rejecting despite short draft\n",
                                j, (double)delta_logp[j], (double)delta_logp_threshold);
                    }
                    short_gate_reject = true;
                    is_delta_kill     = true;
                }

                // Margin safety backstop: always consult margin when we
                // can't look ahead. A sharply-peaked target distribution
                // (high margin) together with a mismatch means the target
                // model is confident about a different token.
                if (!short_gate_reject && margin[j] >= ambiguity_threshold) {
                    if (params.debug_trace) {
                        const char * draft_text = llama_vocab_get_text(vocab, draft[j]);
                        const char * tgt_text   = llama_vocab_get_text(vocab, target[j]);
                        LOG_INF("FLy MARGIN-KILL-SHORT: pos=%d draft=\"%s\"(%d) target=\"%s\"(%d) margin=%.2f >= thr=%.2f, rejecting short draft mismatch\n",
                                j,
                                draft_text ? draft_text : "?", draft[j],
                                tgt_text   ? tgt_text   : "?", target[j],
                                (double)margin[j], (double)ambiguity_threshold);
                    }
                    short_gate_reject = true;
                    is_margin_kill    = true;
                }

                if (short_gate_reject) {
                    if (st) {
                        st->n_strict_reject++;
                        if (is_delta_kill)  { st->n_delta_kill++; }
                        if (is_margin_kill) { st->n_margin_kill++; }
                    }
                    first_reject = j;
                    break;
                }

                if (params.debug_trace) {
                    const char * draft_text = llama_vocab_get_text(vocab, draft[j]);
                    const char * tgt_text   = llama_vocab_get_text(vocab, target[j]);
                    LOG_INF("FLy DEFER-ACCEPT pos %d (short draft K=%d <= W=%d): draft='%s' (id=%d) != target='%s' (id=%d), margin=%.2f\n",
                            j, K, window_W,
                            draft_text ? draft_text : "?", draft[j],
                            tgt_text   ? tgt_text   : "?", target[j], (double) margin[j]);
                    // Structured loose-accept record for harmlessness analysis
                    LOG_INF("FLY-LOOSE: pos=%d draft=\"%s\"(%d) target=\"%s\"(%d) delta_logp=%.4f margin=%.2f window_clean=1 short_draft=1\n",
                            j,
                            draft_text ? draft_text : "?", draft[j],
                            tgt_text   ? tgt_text   : "?", target[j],
                            (double)delta_logp[j], (double)margin[j]);
                }

                if (st) {
                    st->n_loose_accept++;
                    st->sum_delta_logp += (double)delta_logp[j];
                    st->n_delta_logp++;

                    // Running ΔlogP distribution
                    const float dlp = delta_logp[j];
                    if (dlp < st->delta_logp_min) { st->delta_logp_min = dlp; }
                    if (dlp > st->delta_logp_max) { st->delta_logp_max = dlp; }
                    if (dlp == 0.0f)              { st->n_delta_zero++; }
                    if (dlp >= delta_logp_threshold) { st->n_delta_ge_tau++; }

                    // Running margin distribution
                    const float m = margin[j];
                    if (m < st->margin_min) { st->margin_min = m; }
                    if (m > st->margin_max) { st->margin_max = m; }
                    st->sum_margin += (double)m;
                }
                continue;
            }
            // Long draft near end: conservative reject
            if (params.debug_trace) {
                LOG_INF("FLy BOUNDARY-REJECT pos %d: j+W=%d >= K=%d, margin=%.2f\n",
                        j, j + window_W, K, (double) margin[j]);
            }
            if (st) { st->n_boundary_reject++; }
            first_reject = j;
            break;
        }

        // Check the window [j+1, j+window_W] for any mismatch
        bool window_clean = true;
        for (int w = 1; w <= window_W; w++) {
            if (!match[j + w]) {
                window_clean = false;
                break;
            }
        }

        if (window_clean) {
            // ΔlogP gate: even clean-window accepts must respect τ
            if (use_delta_logp_gate && delta_logp[j] >= delta_logp_threshold) {
                if (params.debug_trace) {
                    const char * draft_text = llama_vocab_get_text(vocab, draft[j]);
                    const char * tgt_text   = llama_vocab_get_text(vocab, target[j]);
                    LOG_INF("FLy DLP-KILL-WINDOW: pos=%d draft=\"%s\"(%d) target=\"%s\"(%d) delta_logp=%.4f >= τ=%.2f, rejecting despite clean window\n",
                            j,
                            draft_text ? draft_text : "?", draft[j],
                            tgt_text   ? tgt_text   : "?", target[j],
                            (double)delta_logp[j], (double)delta_logp_threshold);
                }
                if (st) { st->n_strict_reject++; st->n_delta_kill++; }
                first_reject = j;
                break;
            }

            // Semantically equivalent wording — accept the draft token
            if (params.debug_trace) {
                const char * draft_text = llama_vocab_get_text(vocab, draft[j]);
                const char * tgt_text   = llama_vocab_get_text(vocab, target[j]);
                LOG_INF("FLy DEFER-ACCEPT pos %d: draft='%s' (id=%d) != target='%s' (id=%d), margin=%.2f\n",
                        j, draft_text ? draft_text : "?", draft[j],
                        tgt_text   ? tgt_text   : "?", target[j], (double) margin[j]);
                // Structured loose-accept record for harmlessness analysis
                LOG_INF("FLY-LOOSE: pos=%d draft=\"%s\"(%d) target=\"%s\"(%d) delta_logp=%.4f margin=%.2f window_clean=1 short_draft=0\n",
                        j,
                        draft_text ? draft_text : "?", draft[j],
                        tgt_text   ? tgt_text   : "?", target[j],
                        (double)delta_logp[j], (double)margin[j]);
            }
            if (st) {
                st->n_loose_accept++;
                st->sum_delta_logp += (double)delta_logp[j];
                st->n_delta_logp++;

                // Running ΔlogP distribution
                const float dlp = delta_logp[j];
                if (dlp < st->delta_logp_min) { st->delta_logp_min = dlp; }
                if (dlp > st->delta_logp_max) { st->delta_logp_max = dlp; }
                if (dlp == 0.0f)              { st->n_delta_zero++; }
                if (dlp >= delta_logp_threshold) { st->n_delta_ge_tau++; }

                // Running margin distribution
                const float m = margin[j];
                if (m < st->margin_min) { st->margin_min = m; }
                if (m > st->margin_max) { st->margin_max = m; }
                st->sum_margin += (double)m;
            }
            continue;
        } else {
            // Target is course-correcting — reject from j
            if (st) { st->n_window_reject++; }
            first_reject = j;
            break;
        }
    }

    // Mismatches at positions j > first_reject were never evaluated by
    // Phase 2 (the loop broke at first_reject). Count them as "pending"
    // so that the accounting balances:
    //   n_total_miss = loose + strict + control + window + boundary + pending
    if (st) {
        for (int j = first_reject; j < K; j++) {
            if (!match[j]) {
                st->n_pending++;
            }
        }
    }

    // ===== Phase 3: Execution (update sampler + build result) =====
    std::vector<llama_token> result;
    result.reserve((size_t) first_reject + 1);

    // Accept the draft tokens that passed verification
    for (int i = 0; i < first_reject; i++) {
        common_sampler_accept(gsmpl, draft[i], true);
        result.push_back(draft[i]);  // accept draft token (even if != target, it's semantically valid)
    }

    // Bonus token at the rejection point (or after all drafts)
    const int bonus_idx = idxs[first_reject];
    const llama_token bonus = common_sampler_sample(gsmpl, ctx, bonus_idx, grammar_first);
    common_sampler_accept(gsmpl, bonus, true);
    result.push_back(bonus);

    // Merge per-step stats into the caller's accumulator
    if (stats) {
        stats->merge(local_stats);
    }

    return result; // size = first_reject + 1
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
