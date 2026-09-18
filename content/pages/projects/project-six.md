---
type: ProjectLayout
title: Out-of-Order RISC-V CPU
colors: colors-a
date: '2025-12-16'
client: ''
description: >-
  An RV32IM out-of-order core in SystemVerilog built around explicit register
  renaming, a set-associative cache hierarchy, and gshare branch prediction.
---
This is a synthesizable out-of-order processor for the RV32IM instruction set, written in SystemVerilog and verified in Synopsys VCS and Verdi. It was the eight-week final capstone (`mp_ooo`) for ECE 411: Computer Organization & Design at UIUC, built with Pratyush Ashok Anand and Stanley Auyeung under mentor Jason Yan — team "Branch Mispredictors."

An in-order pipeline stalls the moment an instruction waits on a result. An out-of-order core tracks *data dependencies* instead of program order, letting independent instructions issue past a stalled one and retire in order at the end. Building one means building the bookkeeping that makes speculation safe: a rename table, a reorder buffer, reservation stations, a common data bus, and a recovery path for every wrong guess.

What made this project interesting is that it is not graded on speed alone. The course scores designs on `P · D⁴` — power times delay to the fourth — under a hard 300k area ceiling. Almost every decision below is a three-way trade between IPC, area, and power rather than a straight optimization.

### Part 1: Explicit register renaming

We chose explicit register renaming (ERR) early: architectural registers `x0-x31` are decoupled from a larger physical register file, so two instructions writing the same architectural register never falsely serialize. WAW and WAR hazards disappear entirely, leaving only true RAW dependencies for the scheduler to handle, and misprediction recovery stays precise.

The Register Alias Table holds the current logical-to-physical mapping plus a ready bit per row. Rename reads both combinationally, snoops the common data bus to mark results ready as they broadcast, and restores the entire map from the retirement RAT in a single cycle when a flush arrives:

```
always_ff @(posedge clk) begin
  if (rst) begin
    // identity map; x0 is hardwired to p0 and always ready
  end else if (flush_valid) begin
    for (i = 0; i < LOG_REGS; i++) begin
      rat_map[i]   <= rrat_map[i];   // recover from the retirement RAT
      rat_ready[i] <= 1'b1;
    end
  end else begin
    // 1) CDB snoop - mark ready if the row currently points to cdb_prf
    if (cdb_entry.valid) begin
      for (i = 0; i < LOG_REGS; i++) begin
        if (rat_map[i] == cdb_entry.pd)
          rat_ready[i] <= 1'b1;
      end
    end

    // 2) speculative rename write - update mapping and clear Ready
    if (rat_req.rd_alloc && rat_req.alloc_ok) begin
      if (rat_req.rd != '0) begin    // never rename x0
        rat_map  [rat_req.rd] <= rat_req.pd_new;
        rat_ready[rat_req.rd] <= 1'b0;
      end
    end
  end
end
```

Around this sits the rest of the machinery: a free-list FIFO handing out physical registers, reservation station banks (INT ALU, MEM, BRANCH, FP) feeding the functional units, a CDB arbiter resolving simultaneous writebacks, a reorder buffer enforcing in-order commit, and a load-store queue for memory ordering. Issue is age-ordered — each RS entry carries an age field and `pick_ready_oldest` selects the oldest ready entry, with the memory RS forced fully in-order and next-to-commit memory ops given priority so they never get stuck behind another unit.

### Part 2: Branch prediction

Static not-taken was the first thing that had to go. A mispredicted branch costs roughly **6 cycles** of flush against **1 cycle** for a correct prediction, which makes prediction accuracy the dominant front-end lever.

The predictor is gshare: an 8-bit global history register XOR'd with the PC indexes a 256-entry table of 2-bit saturating counters, paired with a 32-entry BTB so a predicted-taken branch can redirect fetch immediately. In the competition branch, the BHT and BTB moved from flip-flop arrays into a compact dual-ported SRAM to cut area.

```
// use word-aligned PC bits as "PC index" (ignore bottom 2 bits) and XOR with GHR
assign bht_idx_out  = (pc[BHT_IDX_BITS+1:2]) ^ ghr_q[BHT_IDX_BITS-1:0];

bht_entry_t ctr_f;
assign ctr_f = bht[bht_idx_out];

// predict taken if MSB of counter is 1
assign predict_taken = ctr_f[1];

always_ff @(posedge clk) begin
  if (rst) begin
    ghr_q <= '0;
    for (i = 0; i < BHT_ENTRIES; i++) bht[i] <= 2'b10;  // weakly taken
  end else begin
    ghr_q <= ghr_d;
    if (update_valid) begin
      if (update_taken) bht[update_bht_idx] <= increment_counter(bht[update_bht_idx]);
      else              bht[update_bht_idx] <= decrement_counter(bht[update_bht_idx]);
    end
  end
end
```

One decision worth calling out: the predictor trains **strictly at commit**, driven by the ROB rather than at execute. Since we were not implementing early branch recovery, commit-time update is the only way the global history survives a flush intact — a speculative update would leave the GHR describing a path the machine never took.

Accuracy was measured in hardware, not estimated: a counter increments on every committed control-flow instruction (branch, JAL, JALR) and a second counter on every detected mispredict, both latched at end of program.

| Benchmark | Control-flow accuracy | Mispredicts / CF instructions |
| --- | --- | --- |
| coremark | 92.7% | 4,527 / 60,470 |
| compression | 92.3% | 5,017 / 65,570 |
| mergesort | 78.9% | 20,702 / 98,078 |
| fft | 77.7% | 7,433 / 33,271 |
| aes_sha | 51.8% | 12,286 / 25,505 |

An 8-entry return address stack handles the call/return pattern a direction predictor gets wrong by construction — pushing on JAL to a link register, popping on JALR returns. On benchmarks it is a small gain (coremark 92.4% → 92.7%), but on a deliberately recursive test program where nearly every JALR was being misclassified as an indirect jump, enabling it took IPC from 0.143 to roughly 0.20 for about 3k of area.

### Part 3: Memory hierarchy

The provided cache was direct-mapped and write-through, and it was the single worst bottleneck in the design — frequently-accessed addresses collided on the same index and evicted each other repeatedly, with no capacity pressure to justify it. Replacing it with a parameterized set-associative cache with tree-PLRU replacement was, by a wide margin, the largest win of the project:

| Benchmark | Direct-mapped | With 4-way SA cache | Change |
| --- | --- | --- | --- |
| compression | 0.211 | 0.483 | +129% |
| fft | 0.216 | 0.446 | +107% |
| mergesort | 0.168 | 0.418 | +148% |
| coremark | 0.170 | 0.424 | +149% |

Two more front-end and back-end additions sit on either side of it. A **streambuffer prefetcher** in front of the I-cache speculatively streams ahead of demand fetches and returns hits in a single cycle, while staying demand-priority so a prefetch never delays a real miss and invalidating itself on any redirect (+9.6% IPC on mergesort, +6.0% on aes_sha, ~0% on compression). A **post-commit store buffer** decouples ROB retirement from the data-cache write path, draining committed stores in the background and forwarding full-word hits to younger loads, with write coalescing merging repeat stores to the same address — about 2.3% average IPC for a 3.1% area increase.

### Trade-offs

The honest version of the cache result is that 4-way, 64-set instruction and data caches passed 4 of 5 benchmarks outright, but pushed total area past the 300k ceiling. Cutting sets from 64 to 8 brought area back in range and IPC down with it, landing just under baseline on most benchmarks and failing aes_sha outright — that workload's access pattern happened to suit the direct-mapped cache better. Taking the benchmark penalty was the cheaper of the two bad options.

Write coalescing was close to free performance-wise (0.054% on one benchmark, nothing elsewhere), which says more about the benchmarks than the feature — these workloads simply do not re-store to the same address often.

Three things we would do differently: split the combined ALU/branch unit so branch resolution does not compete for issue bandwidth, build a fairer I-cache arbiter (the front end appeared to starve during the advanced-features phase), and split the unified LSQ so loads can bypass stores when it is safe to do so.

### For code and the full design writeup:

<https://github.com/siddhshah/RV32IM-OutOfOrder>

An earlier five-stage in-order RV32I pipeline, which preceded this core, lives at <https://github.com/siddhshah/RV32IM-Pipeline>.
