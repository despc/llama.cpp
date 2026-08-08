#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ggml-cpp.h"

struct llama_model;
struct llama_expert_heatmap;

// stores per-layer sizing for the Mixture of Experts GPU hot store.
// one "slot" holds a single expert's weights for one layer.
struct llama_expert_hotstore {
    int n_layers;
    int n_experts;

    // slots requested on the command line (-ehs). Only used to seed the
    // per-device fit; the real counts live in hot_s_layer.
    int hot_s;

    // Slots actually given to each layer. A device sizes its own layers from
    // its own free VRAM, so a card with room is no longer held back by the
    // tightest one. 0 means the layer is not cached.
    std::vector<int> hot_s_layer;

    int slots_of(int il) const {
        return (il >= 0 && il < (int) hot_s_layer.size()) ? hot_s_layer[il] : 0;
    }

    // bytes of a single expert slot per layer, summed over that layer's
    // expert weight tensors (gate/up/down, incl. chexps variants)
    std::vector<size_t> bytes_per_slot;

    // one hot tensor per expert weight tensor, shape {ne0, ne1, hot_s}
    struct entry {
        int          layer_idx;
        ggml_tensor* src; // model tensor holding all n_experts slices
        ggml_tensor* dst; // hot tensor holding hot_s slots
    };
    std::vector<entry> entries;

    // per-layer index into entries (built once in ctor, entries stable after)
    std::vector<std::vector<entry *>> entries_by_layer;

    // slot_to_expert[il][p] = expert id held in slot p of layer il, or -1 if empty.
    // stable across re-syncs: an expert that stays hot keeps its slot.
    std::vector<std::vector<int>> slot_to_expert;

    // per-layer LUT and mask for in-graph routing.
    // hot_lut[e]   = slot index [0..hot_s-1] if e is hot, or hot_s (sentinel) if cold.
    // cold_mask[e] = 1.0f if e is cold, else 0.0f (passed to mul_mat_id_cold).
    struct layer_lut {
        ggml_tensor * hot_lut   = nullptr; // i32[n_experts]
        ggml_tensor * cold_mask = nullptr; // f32[n_experts]
    };
    std::vector<layer_lut> luts; // size n_layers

    // bumped on every resync that swapped >0 slots; build_moe_ffn_tiered
    // compares to its own cached counter to know whether H2D is needed.
    int64_t luts_version = 0;

    // The store is split per device: a layer's slots live on the device that
    // holds the rest of that layer, so the hot path adds no cross-device
    // traffic. One context and one buffer per participating device.
    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // layer_cached[il] is false for a layer whose device cannot host a hot
    // store (CPU, or a non-CUDA backend without --ecf). Those layers keep
    // every expert cold and fall back to the stock mul_mat_id path.
    std::vector<bool> layer_cached;

    // true once the first copy of the top-S experts landed (once per session)
    bool is_filled = false;

    // re-sync cadence in tokens; 0 disables periodic re-sync
    int sync_period = 0;
    // tokens_total at the last sync (fill or re-sync) for boundary-cross check
    int64_t last_sync_tokens = 0;

    // hysteresis gate (Trick 6): a resident slot is only swapped when a cold
    // expert scores >= hyst * the incumbent AND the slot has dwelled long enough
    float hyst  = 0.0f; // 0 = gate off (swap freely)
    int   dwell = 0;    // minimum syncs a resident must keep; 0 = off
    // dwell_count[il][p] = syncs since slot p last changed (0 = fresh/empty)
    std::vector<std::vector<int>> dwell_count;

llama_expert_hotstore(const llama_model * model, int n_layers,
                      int n_experts, int hot_s, int sync_period = 0,
                      float hyst = 0.0f, int dwell = 0);

    ~llama_expert_hotstore();

    // allocate the hot store across the devices that hold the model layers.
    // Each device sizes its own layers from its own free VRAM rather than
    // failing outright. `force` accepts non-CUDA devices (--ecf).
    // returns false (and leaves the store disabled) if nothing could be placed.
    bool allocate(const llama_model * model, bool force);

    // copy the top-S expert slices for every layer into the GPU hot store,
    // using the given heatmap for the ranking. one-shot (guarded by is_filled).
    void copy_top_s(const llama_expert_heatmap & heatmap);

    // static plant: fill slots 0..hot_s-1 with experts 0..hot_s-1 per layer,
    // build LUTs/masks accordingly, no heatmap, no resync. Diagnostic only:
    // isolates the dual-path graph from the heat/dynamic-copy path.
    void plant_static();

    // re-sync the hot store to the current heatmap ranking, swapping only
    // the experts that changed (stable slots; unchanged experts not re-copied).
    void resync_top_s(const llama_expert_heatmap & heatmap);

    // cadence-gated wrapper: re-sync only if tokens_total crossed sync_period;
    // multi_slot freezes the hot store (static slots, no swapping)
    void maybe_resync(const llama_expert_heatmap & heatmap, bool multi_slot);

    // returns the GPU slot index holding expert_id in layer il, or -1 if none
    int slot_of(int layer_idx, int expert_id) const;

    // diagnostic: count how many router-selected expert ids hit a hot slot.
    // reads the selected_experts tensors (call after synchronize).
    void log_hit_rate(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel);

    // rebuild hot_lut/cold_mask from slot_to_expert for every layer
    // and H2D-copy them into the GPU tensors. bumps luts_version.
    // called from copy_top_s (initial fill) and resync_top_s (swaps).
    void update_luts();

    void log() const;
};
