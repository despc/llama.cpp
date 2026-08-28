#include "llama-expert-heatmap.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <vector>

llama_expert_heatmap::llama_expert_heatmap(
        int n_layers, int n_experts,
        float decay_rate, int log_period, int hot_s) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    decay_rate(decay_rate),
    log_period(log_period),
    tokens_total(0),
    heat(n_layers * n_experts, 0.0f) {
}

void llama_expert_heatmap::update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens) {
    float * layer_heat = heat.data() + layer_idx * n_experts;

    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert_used; e++) {
            int32_t id = expert_ids[t * n_expert_used + e];
            if (id >= 0 && id < n_experts) {
                layer_heat[id] += 1.0f;
            }
        }
    }
}
void llama_expert_heatmap::update_from_graph(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel_experts) {
    if (moe_sel_experts.empty()) {
        return;
    }

    decay_all();

    int64_t n_tokens = 0;
    for (const auto & [il, tensor] : moe_sel_experts) {
        n_tokens = tensor->ne[1];

        if (!tensor->data) {
            continue;
        }

        std::vector<int32_t> expert_ids(tensor->ne[0] * n_tokens);
        ggml_backend_tensor_get(tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));

        update(il, expert_ids.data(), tensor->ne[0], n_tokens);
    }

    tokens_total += n_tokens;
    if (n_tokens > 0) {
        sidecar_dirty = true;
    }
    if (log_period > 0 && tokens_total / log_period > (tokens_total - n_tokens) / log_period) {
        log();
    }
}

void llama_expert_heatmap::decay_all() {
    for (int i = 0; i < n_layers * n_experts; i++) {
        heat[i] *= decay_rate;
    }
}

void llama_expert_heatmap::log() const {
    LLAMA_LOG("=== Expert heatmap (tokens %" PRId64 ") ===\n", tokens_total);

    for (int l = 0; l < n_layers; l++) {
        const float * layer_heat = heat.data() + l * n_experts;
        int active_count = 0;
        float max_heat = 0.0f;
        int max_id = -1;

        for (int e = 0; e < n_experts; e++) {
            if (layer_heat[e] > 0.01f) {
                active_count++;
            }
            if (layer_heat[e] > max_heat) {
                max_heat = layer_heat[e];
                max_id = e;
            }
        }

        if (active_count > 0) {
            LLAMA_LOG("  layer %3d: %d warm experts, max heat=%.2f (expert %d)",
                l, active_count, max_heat, max_id);

            auto top = get_top_s(l, 8);
            LLAMA_LOG("  top-8=");
            for (size_t i = 0; i < top.size(); i++) {
                LLAMA_LOG("%s%d", i > 0 ? "," : "{", top[i]);
            }
            LLAMA_LOG("}\n");
        }
    }

    // Share of routed activations captured by the S hottest experts, averaged
    // over layers. Under uniform routing this just tracks S/n_experts, and a
    // hot store of S slots can then do no better than static expert placement.
    // Anything above the uniform column is what an expert cache has to work
    // with.
    const int probes[] = { 32, 64, 96, 128, 256 };
    const int n_probes = (int) (sizeof(probes) / sizeof(probes[0]));

    std::vector<double> captured(n_probes, 0.0);
    std::vector<float>  sorted;
    int n_live = 0;

    for (int l = 0; l < n_layers; l++) {
        const float * layer_heat = heat.data() + l * n_experts;

        double total = 0.0;
        for (int e = 0; e < n_experts; e++) {
            total += layer_heat[e];
        }
        if (total <= 0.0) {
            continue;
        }

        sorted.assign(layer_heat, layer_heat + n_experts);
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        n_live++;

        for (int p = 0; p < n_probes; p++) {
            const int s = std::min(probes[p], n_experts);
            double acc = 0.0;
            for (int i = 0; i < s; i++) {
                acc += sorted[i];
            }
            captured[p] += acc / total;
        }
    }

    if (n_live > 0) {
        LLAMA_LOG("  concentration over %d layers, %d experts:\n", n_live, n_experts);
        for (int p = 0; p < n_probes; p++) {
            const int s = std::min(probes[p], n_experts);
            LLAMA_LOG("    top-%-4d captures %5.1f%%   uniform would be %5.1f%%\n",
                s, 100.0 * captured[p] / n_live, 100.0 * s / n_experts);
        }
    }
}

float llama_expert_heatmap::get_score(int layer_idx, int expert_id) const {
    if (layer_idx < 0 || layer_idx >= n_layers || expert_id < 0 || expert_id >= n_experts) {
        return 0.0f;
    }
    return heat[layer_idx * n_experts + expert_id];
}

std::vector<int> llama_expert_heatmap::get_top_s(int layer_idx, int s) const {
    std::vector<int> result;
    if (layer_idx < 0 || layer_idx >= n_layers || s <= 0) {
        return result;
    }

    const float * layer_heat = heat.data() + layer_idx * n_experts;

    std::vector<int> indices(n_experts);
    for (int i = 0; i < n_experts; i++) {
        indices[i] = i;
    }

    int k = std::min(s, n_experts);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [layer_heat](int a, int b) {
            return layer_heat[a] > layer_heat[b];
        });

    result.assign(indices.begin(), indices.begin() + k);
    return result;
}

// FNV-1a 64-bit, same seed/prime as the wackMall sidecar. Used to fingerprint
// the model so a stale .tier file (different arch, tensor count or shapes)
// is ignored instead of seeding the heatmap with wrong data.
static uint64_t sidecar_fnv1a(const void * data, size_t len, uint64_t h = 14695981039346656037ULL) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t sidecar_fnv1a_str(const char * s, uint64_t h = 14695981039346656037ULL) {
    return sidecar_fnv1a(s, strlen(s), h);
}

static uint64_t sidecar_fnv1a_u64(uint64_t v, uint64_t h = 14695981039346656037ULL) {
    return sidecar_fnv1a(&v, sizeof(v), h);
}

static uint64_t sidecar_fingerprint(const llama_model & model) {
    uint64_t h = 14695981039346656037ULL;
    h = sidecar_fnv1a_str(model.arch_name().c_str(), h);
    h = sidecar_fnv1a_u64((uint64_t) model.size(), h);
    h = sidecar_fnv1a_u64((uint64_t) model.n_tensors(), h);
    for (const auto & kv : model.tensors_by_name) {
        h = sidecar_fnv1a_str(kv.first.c_str(), h);
        const ggml_tensor * t = kv.second;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            h = sidecar_fnv1a_u64((uint64_t) t->ne[i], h);
        }
    }
    return h;
}

void llama_expert_heatmap::init_sidecar(const llama_model & model, bool enable) {
    if (!enable || model.path.empty()) {
        return;
    }

    sidecar_path = model.path + ".tier";
    sidecar_fp   = sidecar_fingerprint(model);

    std::ifstream in(sidecar_path, std::ios::binary);
    if (!in) {
        return;
    }

    uint32_t version = 0;
    uint64_t fp = 0;
    in.read((char *) &version, sizeof(version));
    in.read((char *) &fp, sizeof(fp));
    if (!in || version != 1 || fp != sidecar_fp) {
        LLAMA_LOG("expert sidecar: ignoring %s (version or fingerprint mismatch)\n", sidecar_path.c_str());
        return;
    }

    const size_t want = (size_t) n_layers * (size_t) n_experts;
    std::vector<float> loaded(want);
    in.read((char *) loaded.data(), want * sizeof(float));
    if (!in || (size_t) in.gcount() != want * sizeof(float)) {
        LLAMA_LOG("expert sidecar: ignoring %s (truncated)\n", sidecar_path.c_str());
        return;
    }

    heat = std::move(loaded);
    sidecar_dirty = false;
    LLAMA_LOG("expert sidecar: loaded %s (%zu layers x %d experts)\n",
        sidecar_path.c_str(), (size_t) n_layers, n_experts);
}

void llama_expert_heatmap::save_sidecar() const noexcept {
    if (sidecar_path.empty() || !sidecar_dirty || sidecar_fp == 0) {
        return;
    }

    const std::string tmp = sidecar_path + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
        return;
    }

    const uint32_t version = 1;
    out.write((const char *) &version, sizeof(version));
    out.write((const char *) &sidecar_fp, sizeof(sidecar_fp));
    out.write((const char *) heat.data(), heat.size() * sizeof(float));
    out.close();

    if (!out || std::rename(tmp.c_str(), sidecar_path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return;
    }

    LLAMA_LOG("expert sidecar: saved %s (%zu layers x %d experts)\n",
        sidecar_path.c_str(), (size_t) n_layers, n_experts);
}

llama_expert_heatmap::~llama_expert_heatmap() {
    save_sidecar();
}
