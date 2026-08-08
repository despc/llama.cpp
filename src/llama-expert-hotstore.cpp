#include "llama-expert-hotstore.h"
#include "llama-expert-heatmap.h"
#include "llama-expert-tier.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstring>
#include <regex>

// matches the weight tensor of an expert tensor, e.g.:
//   blk.0.ffn_gate_exps.weight
//   blk.3.ffn_down_chexps.weight
// follows the same convention as LLM_FFN_EXPS_REGEX in common.h
static const std::regex g_re_exps_weight("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

llama_expert_hotstore::llama_expert_hotstore(
        const llama_model * model, int n_layers, int n_experts, int hot_s, int sync_period,
        float hyst, int dwell) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    bytes_per_slot(n_layers, 0),
    sync_period(sync_period),
    hyst(hyst),
    dwell(dwell) {
    if (n_layers <= 0) {
        return;
    }

    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        std::smatch m;
        if (std::regex_search(name, m, g_re_exps_weight)) {
            const int il = std::stoi(m[1].str());
            if (il >= 0 && il < n_layers && tensor->ne[2] > 0) {
                // a slot holds nbytes/n_experts of this tensor
                bytes_per_slot[il] += ggml_nbytes(tensor) / (size_t) tensor->ne[2];
                entries.push_back({il, tensor, nullptr});
            }
        }
    }

    // entries is fixed from here on; build a per-layer index of stable
    // pointers so copy/resync do not iterate the whole entries vector.
    entries_by_layer.assign(n_layers, {});
    for (auto & e : entries) {
        entries_by_layer[e.layer_idx].push_back(&e);
    }

    // per-layer slot counts are decided in allocate(), once the devices and
    // their free VRAM are known
    hot_s_layer.assign(n_layers, 0);
    slot_to_expert.assign(n_layers, {});
    dwell_count.assign(n_layers, {});
}

// VRAM left untouched on every participating device, so sizing the store does
// not eat the headroom the compute buffers still need. 512 MiB was not enough:
// the graph reserve that runs after this allocation then faulted inside the
// driver rather than reporting a clean allocation failure.
static size_t hotstore_dev_margin() {
    return 2048ull * 1024 * 1024;
}

bool llama_expert_hotstore::allocate(const llama_model * model, bool force) {
    if (hot_s <= 0 || entries.empty() || model == nullptr) {
        return false;
    }
    if (hot_s > n_experts) {
        // asking for more slots than the model has experts is a request to
        // cache all of them, not an error - the same command line then works
        // across models with different expert counts
        LLAMA_LOG_INFO("%s: hot store: %d slots requested, model has %d experts, using %d\n",
            __func__, hot_s, n_experts, n_experts);
        hot_s = n_experts;
    }

    layer_cached.assign(n_layers, false);

    // Group the layers that own expert tensors by the device holding the rest
    // of that layer. A layer's slots must sit on its own device: otherwise the
    // hot path pulls the activations across the bus once per layer per token,
    // which on a chipset-attached card costs more than the DDR read it saves.
    std::vector<ggml_backend_dev_t>   devs;
    std::vector<std::vector<int>>     dev_layers;

    for (int il = 0; il < n_layers; il++) {
        if (entries_by_layer[il].empty()) {
            continue;
        }
        // Only a layer whose experts sit in host memory has anything to gain.
        // A layer already resident on a GPU is read at VRAM bandwidth, so
        // caching it would add the dual hot/cold path - and its per-layer cost
        // - for no saving at all.
        const ggml_tensor * probe = entries_by_layer[il][0]->src;
        if (probe->buffer == nullptr || !ggml_backend_buffer_is_host(probe->buffer)) {
            continue;
        }
        ggml_backend_dev_t dev = model->dev_layer(il);
        if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        // the dual-path graph is only validated on CUDA; on other backends it
        // either buys nothing (CPU) or corrupts output (Vulkan). Note this
        // matches "V100_CUDA" as well as "CUDA".
        const char * dname = ggml_backend_dev_name(dev);
        if (!force && (dname == nullptr || strstr(dname, "CUDA") == nullptr)) {
            continue;
        }
        size_t id = 0;
        while (id < devs.size() && devs[id] != dev) {
            id++;
        }
        if (id == devs.size()) {
            devs.push_back(dev);
            dev_layers.push_back({});
        }
        dev_layers[id].push_back(il);
    }

    if (devs.empty()) {
        LLAMA_LOG_WARN("%s: hot store: no eligible device holds a MoE layer\n", __func__);
        return false;
    }

    // Each device sizes its own layers from its own free VRAM. Slot indices are
    // per layer, so a card with room is not held back by the tightest one - a
    // 32 GiB V100 holding few layers can cache far more of each than a full
    // 16 GiB card next to it.
    const size_t lut_bytes = (size_t) n_experts * (sizeof(int32_t) + sizeof(float));
    const size_t margin    = hotstore_dev_margin();

    bool any = false;
    for (size_t id = 0; id < devs.size(); id++) {
        size_t per_slot = 0;
        for (int il : dev_layers[id]) {
            per_slot += bytes_per_slot[il];
        }
        if (per_slot == 0) {
            continue;
        }
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(devs[id], &free_mem, &total_mem);

        const size_t overhead = lut_bytes * dev_layers[id].size() + margin;
        const size_t usable   = free_mem > overhead ? free_mem - overhead : 0;
        // one sentinel slot per layer is always allocated on top of the count
        const size_t n_slot   = usable / per_slot;
        int dev_s = n_slot > 1 ? (int) (n_slot - 1) : 0;

        dev_s = std::min(dev_s, hot_s);
        dev_s = std::min(dev_s, n_experts);

        if (dev_s <= 0) {
            LLAMA_LOG_WARN("%s: hot store: no room on %s, its %zu layers stay cold\n",
                __func__, ggml_backend_dev_name(devs[id]), dev_layers[id].size());
            continue;
        }

        for (int il : dev_layers[id]) {
            hot_s_layer[il]    = dev_s;
            slot_to_expert[il] = std::vector<int>(dev_s, -1);
            dwell_count[il]    = std::vector<int>(dev_s, 0);
        }
        any = true;
    }

    if (!any) {
        LLAMA_LOG_WARN("%s: hot store: not enough free VRAM for even one slot\n", __func__);
        return false;
    }

    luts.assign(n_layers, layer_lut{});
    ctxs.resize(devs.size());
    bufs.resize(devs.size());

    for (size_t id = 0; id < devs.size(); id++) {
        const size_t n_ent = [&] {
            size_t n = 0;
            for (int il : dev_layers[id]) {
                n += entries_by_layer[il].size();
            }
            return n;
        }();

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (n_ent + 2 * dev_layers[id].size() + 4),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctxs[id] = ggml_context_ptr(ggml_init(params));
        if (!ctxs[id]) {
            LLAMA_LOG_ERROR("%s: hot store: failed to create ggml context for %s\n",
                __func__, ggml_backend_dev_name(devs[id]));
            return false;
        }

        // one hot tensor per model expert tensor, with the layer's slot count
        // plus 1 sentinel slot (index slots_of(il)) that stays zero so cold
        // selections read zeros via a valid in-range index (oldtricks Trick 2).
        for (int il : dev_layers[id]) {
            if (slots_of(il) <= 0) {
                continue;
            }
            for (entry * e : entries_by_layer[il]) {
                e->dst = ggml_new_tensor_3d(ctxs[id].get(), e->src->type,
                    e->src->ne[0], e->src->ne[1], slots_of(il) + 1);
            }
            // per-layer LUT and mask for in-graph routing (oldtricks Trick 4)
            luts[il].hot_lut   = ggml_new_tensor_1d(ctxs[id].get(), GGML_TYPE_I32, n_experts);
            luts[il].cold_mask = ggml_new_tensor_1d(ctxs[id].get(), GGML_TYPE_F32, n_experts);
        }

        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(devs[id]);
        const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctxs[id].get(), buft);
        if (need == 0) {
            LLAMA_LOG_ERROR("%s: hot store: zero-sized buffer on %s, disabled\n",
                __func__, ggml_backend_dev_name(devs[id]));
            return false;
        }

        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctxs[id].get(), buft);
        if (b == nullptr) {
            throw std::runtime_error(format("%s: unable to allocate hot store on %s (%zu MiB)",
                __func__, ggml_backend_dev_name(devs[id]), need / (1024 * 1024)));
        }
        bufs[id] = ggml_backend_buffer_ptr(b);
        ggml_backend_buffer_set_usage(bufs[id].get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // zero the whole buffer so the sentinel slot (index hot_s) AND every
        // not-yet-filled expert slot is zero; copy_top_s/resync_top_s only write
        // slots 0..hot_s-1, so slot hot_s stays zero for the lifetime of the store.
        ggml_backend_buffer_clear(bufs[id].get(), 0);

        // register the expert weight tensors with the tier hook so
        // build_lora_mm_id can find the hot tensor and the per-layer LUTs.
        int n_cached_here = 0;
        for (int il : dev_layers[id]) {
            if (slots_of(il) <= 0) {
                continue;
            }
            for (const entry * e : entries_by_layer[il]) {
                llama_expert_tier_register(e->src, e->dst, luts[il].hot_lut, luts[il].cold_mask);
            }
            layer_cached[il] = true;
            n_cached_here++;
        }

        LLAMA_LOG("  hot store on %s: %zu MiB for %d layers, %d+1 slots each\n",
            ggml_backend_dev_name(devs[id]), need / (1024 * 1024),
            n_cached_here, slots_of(dev_layers[id][0]));
    }

    // Mark every expert cold before the first fill. The LUT buffers are zeroed
    // above, and zero means "hot, slot 0" - which routes every expert to an
    // empty slot and tells the CPU path to skip it, so the routed experts
    // contribute nothing until the first copy_top_s lands. That window used to
    // be a single token; it is as long as fill_delay now, and it degrades the
    // output either way.
    update_luts();

    return true;
}

llama_expert_hotstore::~llama_expert_hotstore() {
    llama_expert_tier_clear();
}

void llama_expert_hotstore::copy_top_s(const llama_expert_heatmap & heatmap) {
    if (is_filled || entries.empty() || bufs.empty()) {
        return;
    }

    for (int il = 0; il < n_layers; il++) {
        if (!layer_cached[il]) {
            continue;
        }
        const int S = slots_of(il);
        const std::vector<int> top = heatmap.get_top_s(il, S);
        auto & ste = slot_to_expert[il];
        auto & dc  = dwell_count[il];
        for (int p = 0; p < (int) top.size() && p < S; p++) {
            ste[p] = top[p];
            dc[p]  = dwell; // initial fill is eligible to be corrected next sync
        }

        for (entry * e : entries_by_layer[il]) {
            const size_t slot = ggml_nbytes(e->src) / (size_t) e->src->ne[2];
            const char * src = e->src->data ? (const char *) ggml_get_data(e->src) : nullptr;
            if (!src) {
                continue;
            }
            for (int p = 0; p < S; p++) {
                const int ex = ste[p];
                if (ex < 0) {
                    continue;
                }
                ggml_backend_tensor_set(e->dst, src + (size_t) ex * slot, (size_t) p * slot, slot);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    is_filled = true;
    update_luts();
    LLAMA_LOG("=== Expert hot store: top-S experts copied to GPU ===\n");
}

void llama_expert_hotstore::plant_static() {
    if (is_filled || entries.empty() || bufs.empty()) {
        return;
    }

    // plant experts 0..S-1 into slots 0..S-1, one shot, no heatmap
    for (int il = 0; il < n_layers; il++) {
        if (!layer_cached[il]) {
            continue;
        }
        const int S = slots_of(il);
        auto & ste = slot_to_expert[il];
        for (int p = 0; p < S && p < n_experts; p++) {
            ste[p] = p;
        }
        for (entry * e : entries_by_layer[il]) {
            const size_t slot = ggml_nbytes(e->src) / (size_t) e->src->ne[2];
            const char * src = e->src->data ? (const char *) ggml_get_data(e->src) : nullptr;
            if (!src) continue;
            for (int p = 0; p < S && p < n_experts; p++) {
                ggml_backend_tensor_set(e->dst, src + (size_t) p * slot, (size_t) p * slot, slot);
            }
        }
    }

    is_filled = true;
    update_luts();
    LLAMA_LOG("=== Expert hot store: STATIC plant done ===\n");
}

void llama_expert_hotstore::resync_top_s(const llama_expert_heatmap & heatmap) {
    if (!is_filled || bufs.empty()) {
        return;
    }

    // tokens elapsed since the previous sync, used to age dwell counters
    const int64_t elapsed = heatmap.tokens_total - last_sync_tokens;
    int swapped = 0;
    for (int il = 0; il < n_layers; il++) {
        if (!layer_cached[il]) {
            continue;
        }
        const int S = slots_of(il);
        const std::vector<int> top = heatmap.get_top_s(il, S);
        auto & ste = slot_to_expert[il];
        auto & dc  = dwell_count[il];

        // find a free slot index (guard: any resident displacement must
        // clear the hysteresis gate, unless the gate is off)
        auto find_slot = [&](int e_cold) -> int {
            // fill empty slots first (no gate on fill)
            for (int p = 0; p < S; p++) {
                if (ste[p] < 0) {
                    return p;
                }
            }
            if (hyst <= 0.0f) {
                // gate off: displace the weakest resident
                int p_worst = -1;
                for (int p = 0; p < S; p++) {
                    if (ste[p] >= 0 && (p_worst < 0 ||
                        heatmap.get_score(il, ste[p]) < heatmap.get_score(il, ste[p_worst]))) {
                        p_worst = p;
                    }
                }
                return p_worst;
            }
            // gate on: coldest resident that has dwelled enough AND is beaten
            // by hyst * this cold expert
            const float s_cold = heatmap.get_score(il, e_cold);
            int p_worst = -1;
            float worst_score = 1e9f;
            for (int p = 0; p < S; p++) {
                if (ste[p] < 0) {
                    continue;
                }
                if (dc[p] < dwell) {
                    continue; // incumbent must keep its slot (Trick 6)
                }
                if (s_cold >= hyst * heatmap.get_score(il, ste[p])) {
                    const float s_inc = heatmap.get_score(il, ste[p]);
                    if (s_inc < worst_score) {
                        worst_score = s_inc;
                        p_worst     = p;
                    }
                }
            }
            return p_worst;
        };

        // resident experts -> candidate cold experts, most significant first
        std::vector<char> resident_set(n_experts, 0);
        for (int p = 0; p < S; p++) {
            if (ste[p] >= 0) {
                resident_set[ste[p]] = 1;
            }
        }
        for (int e_cold : top) {
            if (e_cold < 0 || e_cold >= n_experts || resident_set[e_cold]) {
                continue;
            }
            const int p = find_slot(e_cold);
            if (p < 0) {
                break; // no slot free or displaceable under the gate
            }
            for (entry * ent : entries_by_layer[il]) {
                const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                const char * src = ent->src->data ? (const char *) ggml_get_data(ent->src) : nullptr;
                if (!src) {
                    continue;
                }
                ggml_backend_tensor_set(ent->dst, src + (size_t) e_cold * slot, (size_t) p * slot, slot);
            }
            ste[p] = e_cold;
            dc[p]  = -elapsed; // fresh dwell: aging below brings it to 0
            swapped++;
        }

        for (int p = 0; p < S; p++) {
            if (ste[p] >= 0) {
                dc[p] += (int) std::max<int64_t>(elapsed, 0);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    if (swapped > 0) {
        update_luts();
        if (getenv("LLAMA_EXPERT_DEBUG")) {
            LLAMA_LOG("=== Expert hot store: re-sync swapped %d expert slots ===\n", swapped);
        }
    }
}

void llama_expert_hotstore::maybe_resync(const llama_expert_heatmap & heatmap, bool multi_slot) {
    // n_tokens>1 (multi-slot) freezes the hot store: no swapping during the batch
    if (multi_slot || sync_period <= 0 || heatmap.tokens_total <= 0) {
        return;
    }
    if (heatmap.tokens_total / sync_period > last_sync_tokens / sync_period) {
        resync_top_s(heatmap);
    }
}

int llama_expert_hotstore::slot_of(int layer_idx, int expert_id) const {
    const int S = slots_of(layer_idx);
    if (layer_idx < 0 || layer_idx >= n_layers || S <= 0) {
        return -1;
    }
    const auto & ste = slot_to_expert[layer_idx];
    for (int p = 0; p < S; p++) {
        if (ste[p] == expert_id) {
            return p;
        }
    }
    return -1;
}

void llama_expert_hotstore::update_luts() {
    if (luts.empty() || bufs.empty()) {
        return;
    }

    std::vector<int32_t> hot_lut_h(n_experts);
    std::vector<float>   cold_mask_h(n_experts);

    for (int il = 0; il < n_layers; il++) {
        if (!layer_cached[il] || luts[il].hot_lut == nullptr) {
            continue;
        }
        const int S = slots_of(il);
        const auto & ste = slot_to_expert[il];

        // defaults: everyone cold
        for (int e = 0; e < n_experts; e++) {
            hot_lut_h[e]   = S;         // sentinel slot (zero)
            cold_mask_h[e] = 1.0f;
        }

        // residents override
        for (int p = 0; p < S; p++) {
            const int e = ste[p];
            if (e < 0) {
                continue;
            }
            hot_lut_h[e]   = p;         // its slot index
            cold_mask_h[e] = 0.0f;
        }

        const size_t bytes_i32 = n_experts * sizeof(int32_t);
        const size_t bytes_f32 = n_experts * sizeof(float);
        ggml_backend_tensor_set(luts[il].hot_lut,   hot_lut_h.data(),   0, bytes_i32);
        ggml_backend_tensor_set(luts[il].cold_mask, cold_mask_h.data(), 0, bytes_f32);
    }

    luts_version++;
}

void llama_expert_hotstore::log_hit_rate(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel) {
    if (moe_sel.empty() || !is_filled) {
        return;
    }
    size_t hits = 0, total = 0;
    for (const auto & kv : moe_sel) {
        const int il = kv.first;
        const ggml_tensor * t = kv.second;
        if (!t || !t->data || t->type != GGML_TYPE_I32) {
            continue;
        }
        const size_t n = ggml_nelements(t);
        std::vector<int32_t> ids(n);
        ggml_backend_tensor_get(t, ids.data(), 0, n * sizeof(int32_t));
        for (size_t i = 0; i < n; i++) {
            const int32_t id = ids[i];
            if (id >= 0 && id < n_experts) {
                total++;
                if (slot_of(il, id) >= 0) {
                    hits++;
                }
            }
        }
    }
    if (total > 0) {
        LLAMA_LOG("=== expert hot hit rate: %zu/%zu = %.1f%% ===\n", hits, total, 100.0f * (float) hits / (float) total);
    }
}

void llama_expert_hotstore::log() const {
    LLAMA_LOG("=== Expert hotstore sizing (S=%d) ===\n", hot_s);
    size_t total = 0;
    for (int il = 0; il < n_layers; il++) {
        total += bytes_per_slot[il];
        LLAMA_LOG("  layer %3d: bytes/slot = %zu\n", il, bytes_per_slot[il]);
    }
    LLAMA_LOG("  total bytes/slot across all layers = %zu (%zu MiB)\n",
        total, total / (1024 * 1024));
    if (!bufs.empty()) {
        size_t allocated = 0;
        int    n_cached  = 0;
        for (const auto & b : bufs) {
            if (b) {
                allocated += ggml_backend_buffer_get_size(b.get());
            }
        }
        for (int il = 0; il < n_layers; il++) {
            n_cached += layer_cached[il] ? 1 : 0;
        }
        LLAMA_LOG("  GPU hot store allocated: %zu MiB over %zu device(s), %d layers cached, %d+1 slots\n",
            allocated / (1024 * 1024), bufs.size(), n_cached, hot_s);
    } else if (hot_s > 0) {
        LLAMA_LOG("  hot store DISABLED (%d slots requested)\n", hot_s);
    }
}
