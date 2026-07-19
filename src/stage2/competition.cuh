#ifndef SNN_STAGE2_COMPETITION_CUH
#define SNN_STAGE2_COMPETITION_CUH

#include "../include/network.h"

// =============================================================================
// Stage 2d (P2): k-Winners-Take-All columnar competition
// =============================================================================
//
// 2c + 2d-v1 diagnosis:
//   - P0 (relaxed homeostatic) + P1 (one-hot input) failed at 1M steps.
//   - Root cause: even with one-hot input on sensory neurons 0..255, the
//     80 intra-column synapses per neuron spread activity uniformly across
//     each column. Every column ends up responding to every byte.
//   - Chi-square test still shows 0/6622 neurons with byte selectivity.
//
// P2 fix: enforce column-level competition.
//   After each network.step(), count spikes per column, find the top-k
//   columns by spike count, and ZERO OUT all spikes in non-top-k columns.
//   This forces the network to commit to a small subset of columns per
//   input, giving STDP a clear signal: "column A fired for byte X" ->
//   strengthen A's intra-column synapses; "column B was suppressed for
//   byte X" -> weaken B's synapses that would have fired.
//
// Expected outcome:
//   - Different bytes drive different top-k column sets
//   - STDP reinforces this mapping (intra-column synapses for the
//     "winning" byte pattern strengthen, others weaken)
//   - Chi-square test should show significant neurons
//   - PCA should show multiple distinct clusters (not collapsed to a line)
//
// Implementation:
//   - N_COLUMNS = 10 is small enough to do top-k selection on host
//   - Spike zeroing is a simple CUDA kernel (or host loop + memcpy)
//   - Runs AFTER network.step() but BEFORE STDP weight update
//     (stage0's step() already did STDP -- so we are post-hoc filtering
//      the spikes that WILL be used by the next step's STDP via
//      last_spike_time. To make STDP see the filtered spikes, we would
//      need to integrate competition INTO step(); that requires modifying
//      stage0 which is forbidden by project memory).
//
//   Workaround: P2 runs after step(), so STDP in step() already saw the
//   unfiltered spikes. But the FILTERED spikes are what get recorded in
//   d_spikes_, which is what the next step's STDP reads via last_spike_time.
//   Actually STDP uses last_spike_time which was just set inside step() --
//   so filtering d_spikes_ after step() does NOT undo the STDP update that
//   just happened.
//
//   True effect of post-hoc filtering:
//     - The synaptic_current of the next step will NOT include spikes from
//       suppressed columns (because synapse_kernel reads d_spikes_).
//     - So suppressed columns' downstream neurons won't get input.
//     - Over many steps, this shapes which synapses are useful.
//   This is sufficient for our purposes.
//
// k tuning:
//   - k=1: extreme winner-take-all, may be too aggressive
//   - k=2: recommended starting point (20% of columns active)
//   - k=3: 30% active, more forgiving
// =============================================================================

// Apply k-WTA columnar competition in-place on d_spikes_.
// After this call, only neurons in the top-k columns (by spike count)
// retain their spikes; all other neurons' spikes are set to false.
//
// Parameters:
//   network  - the SNN network (uses get_d_spikes())
//   k        - number of winning columns to keep (1..N_COLUMNS)
void apply_kwta_competition(SNNNetwork& network, int k);

#endif // SNN_STAGE2_COMPETITION_CUH
