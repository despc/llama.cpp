#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <utility>

struct ggml_tensor;
struct llama_model;

struct llama_expert_heatmap {
    int n_layers;
    int n_experts;
    int hot_s;
    float decay_rate;
    int   log_period;
    int64_t tokens_total; // real tokens seen (not multiplied by layers)

    std::vector<float> heat;

    llama_expert_heatmap(int n_layers, int n_experts,
                         float decay_rate = 0.99f,
                         int log_period = 100,
                         int hot_s = 0);
    ~llama_expert_heatmap();

    void update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens);
    void update_from_graph(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel_experts);
    void decay_all();
    void log() const;

    float get_score(int layer_idx, int expert_id) const;
    std::vector<int> get_top_s(int layer_idx, int s) const;

    // warm-start sidecar: persist the converged heat set to <model>.tier on
    // shutdown, reload it on the next run so a restart skips the convergence
    // burst. init_sidecar() computes a fingerprint from the model and loads the
    // file if it matches; the destructor saves it back when heat has changed.
    void init_sidecar(const llama_model & model, bool enable);

private:
    std::string sidecar_path; // empty when sidecar is disabled
    uint64_t    sidecar_fp   = 0;
    bool        sidecar_dirty = false;

    void save_sidecar() const noexcept;
};
