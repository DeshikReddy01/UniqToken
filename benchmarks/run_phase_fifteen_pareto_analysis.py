"""
Phase 15: Compute-Normalized Pareto Frontier & Multi-Objective Decision Analysis
================================================================================
Constructs the multi-objective Pareto surface across:
    (Vocabulary Scale V, LM Capacity Tier, Tokenizer Architecture, Total Parameters P)
in the objective space:
    (True LM BPB, Token Cross-Entropy CE Loss)

Performs:
1. Exact Pareto Non-Dominated Set Extraction in (BPB, CE) space (Global and Tier-by-Tier).
2. Global and Tier-Specific argmin(BPB) and argmin(CE) identification.
3. Constrained Optimization:
   - min BPB s.t. CE <= tau_CE (for tau_CE in [9.5, 10.0, 10.5, 11.0, 11.2, 11.5, 12.0, 12.5])
   - min CE s.t. BPB <= tau_BPB (for tau_BPB in [2.0, 2.3, 2.5, 2.7, 2.9, 3.1, 3.4])
   - Parameter-Constrained Optimization (P <= 20M, P <= 50M)
4. Regime Boundary Characterization (Where does each tokenizer dominate?).
5. Publication-grade 4-panel Pareto Frontier Visualization.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_all_records():
    benchmarks_dir = r"c:\Users\shaik\Research\Tokenizer\benchmarks"
    confirmatory_path = os.path.join(benchmarks_dir, "phase_fourteen_confirmatory_records.json")
    capacity_path = os.path.join(benchmarks_dir, "phase_fourteen_capacity_records.json")
    
    with open(confirmatory_path, "r", encoding="utf-8") as f:
        conf_data = json.load(f)
    with open(capacity_path, "r", encoding="utf-8") as f:
        cap_data = json.load(f)
        
    dataset = []
    
    # 1. Load 16K data from Phase 14A (mean across 3 seeds)
    tiers = ["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"]
    tokenizers = ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]
    
    for tier in tiers:
        for tok in tokenizers:
            entry_16k = cap_data["summary_grid"][tier]["16384"][tok]
            dataset.append({
                "tier": tier,
                "vocab_size": 16384,
                "vocab_label": "16K",
                "tokenizer": tok,
                "bpb": float(entry_16k["true_lm_bpb"]),
                "ce": float(entry_16k["token_ce_loss"]),
                "bytes_per_token": float(entry_16k["bytes_per_token"]),
                "active_vocab_pct": float(entry_16k["active_vocab_pct"]),
                "total_params": int(entry_16k["total_params"]),
                "training_steps": int(entry_16k["training_steps"]),
                "source": "Phase 14A (N=3)"
            })
            
    # 2. Load 32K and 64K data from Phase 14B (mean across 5 seeds)
    for tier in tiers:
        for v_scale, v_label in [("32768", "32K"), ("65536", "64K")]:
            for tok in tokenizers:
                entry = conf_data["summary_grid"][tier][v_scale][tok]
                ref_entry = cap_data["summary_grid"][tier][v_scale][tok]
                dataset.append({
                    "tier": tier,
                    "vocab_size": int(v_scale),
                    "vocab_label": v_label,
                    "tokenizer": tok,
                    "bpb": float(entry["true_lm_bpb_mean"]),
                    "bpb_std": float(entry["true_lm_bpb_std"]),
                    "ce": float(entry["token_ce_loss_mean"]),
                    "bytes_per_token": float(entry["bytes_per_token_mean"]),
                    "active_vocab_pct": float(entry["active_vocab_pct_mean"]),
                    "total_params": int(ref_entry["total_params"]),
                    "training_steps": int(ref_entry["training_steps"]),
                    "source": "Phase 14B (N=5)"
                })
                
    return dataset

def compute_pareto_frontier(subset):
    """
    A point A dominates point B if:
    bpb_A <= bpb_B and ce_A <= ce_B and (bpb_A < bpb_B or ce_A < ce_B).
    Returns the list of non-dominated Pareto points sorted by BPB ascending.
    """
    pareto_points = []
    for i, candidate in enumerate(subset):
        is_dominated = False
        for j, other in enumerate(subset):
            if i == j:
                continue
            if (other["bpb"] <= candidate["bpb"] and other["ce"] <= candidate["ce"]) and \
               (other["bpb"] < candidate["bpb"] or other["ce"] < candidate["ce"]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(candidate)
            
    pareto_points.sort(key=lambda x: x["bpb"])
    return pareto_points

def run_constrained_decision_analysis(dataset):
    """
    Evaluates:
    1. min BPB s.t. CE <= tau_CE
    2. min CE s.t. BPB <= tau_BPB
    3. min BPB s.t. Params <= tau_Params
    """
    ce_thresholds = [9.5, 10.0, 10.5, 11.0, 11.2, 11.5, 12.0, 12.5]
    bpb_thresholds = [2.0, 2.3, 2.5, 2.7, 2.9, 3.1, 3.4]
    param_thresholds = [10.0, 20.0, 40.0, 60.0, 100.0]  # Millions
    
    ce_constrained_results = []
    for tau in ce_thresholds:
        valid = [d for d in dataset if d["ce"] <= tau]
        if valid:
            best = min(valid, key=lambda x: x["bpb"])
            ce_constrained_results.append({
                "ce_ceiling": tau,
                "best_bpb": best["bpb"],
                "best_ce": best["ce"],
                "tokenizer": best["tokenizer"],
                "tier": best["tier"],
                "vocab": best["vocab_label"],
                "total_params": best["total_params"]
            })
        else:
            ce_constrained_results.append({
                "ce_ceiling": tau,
                "best_bpb": None,
                "tokenizer": "Infeasible",
                "tier": "-",
                "vocab": "-"
            })
            
    bpb_constrained_results = []
    for tau in bpb_thresholds:
        valid = [d for d in dataset if d["bpb"] <= tau]
        if valid:
            best = min(valid, key=lambda x: x["ce"])
            bpb_constrained_results.append({
                "bpb_ceiling": tau,
                "best_ce": best["ce"],
                "best_bpb": best["bpb"],
                "tokenizer": best["tokenizer"],
                "tier": best["tier"],
                "vocab": best["vocab_label"],
                "total_params": best["total_params"]
            })
        else:
            bpb_constrained_results.append({
                "bpb_ceiling": tau,
                "best_ce": None,
                "tokenizer": "Infeasible",
                "tier": "-",
                "vocab": "-"
            })
            
    param_constrained_results = []
    for tau in param_thresholds:
        valid = [d for d in dataset if (d["total_params"] / 1e6) <= tau]
        if valid:
            best = min(valid, key=lambda x: x["bpb"])
            param_constrained_results.append({
                "param_ceiling_m": tau,
                "best_bpb": best["bpb"],
                "best_ce": best["ce"],
                "tokenizer": best["tokenizer"],
                "tier": best["tier"],
                "vocab": best["vocab_label"],
                "total_params": best["total_params"]
            })
            
    return ce_constrained_results, bpb_constrained_results, param_constrained_results

def plot_pareto_frontier(dataset, pareto_points, ce_constrained, out_path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    
    tok_colors = {
        "SentencePiece-Unigram": "#2b5c8f",
        "Boundary-BPE": "#d95f02",
        "Caliper-SuperBPE (Config B)": "#7570b3"
    }
    tier_markers = {
        "Small (4L-128d)": "o",
        "Medium (6L-256d)": "s",
        "Large (8L-512d)": "^"
    }
    
    # -------------------------------------------------------------
    # Panel 1: Full 2D Pareto Objective Space (BPB vs CE)
    # -------------------------------------------------------------
    ax1 = axs[0, 0]
    
    for tok in tok_colors:
        sub = [d for d in dataset if d["tokenizer"] == tok]
        x = [d["ce"] for d in sub]
        y = [d["bpb"] for d in sub]
        ax1.scatter(x, y, color=tok_colors[tok], label=tok.split("-")[0], alpha=0.6, s=70)
        
    for d in dataset:
        ax1.scatter(d["ce"], d["bpb"], color=tok_colors[d["tokenizer"]], 
                    marker=tier_markers[d["tier"]], s=110, edgecolors="black", linewidths=0.7)
        if d["vocab_label"] == "64K" and d["tier"] == "Large (8L-512d)":
            ax1.annotate(f"{d['tokenizer'].split('-')[0]} 64K-L", (d["ce"], d["bpb"]),
                         xytext=(6, 4), textcoords="offset points", fontsize=9, fontweight="bold",
                         color=tok_colors[d["tokenizer"]])
        elif d["vocab_label"] == "32K" and d["tier"] == "Large (8L-512d)":
            ax1.annotate(f"{d['tokenizer'].split('-')[0]} 32K-L", (d["ce"], d["bpb"]),
                         xytext=(6, -10), textcoords="offset points", fontsize=8,
                         color=tok_colors[d["tokenizer"]])

    # Plot Pareto Frontier Curve
    px = [p["ce"] for p in pareto_points]
    py = [p["bpb"] for p in pareto_points]
    ax1.plot(px, py, color="#e41a1c", linestyle="--", linewidth=2.2, label="Global Pareto Frontier")
    ax1.scatter(px, py, color="#e41a1c", marker="*", s=260, edgecolors="black", linewidths=1.2, zorder=5, label="Pareto-Optimal Points")
    
    ax1.set_xlabel("Token Cross-Entropy Loss (nats) -> min", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True LM BPB -> min", fontsize=12, fontweight="bold")
    ax1.set_title("A: Multi-Objective Tradeoff & Global Pareto Frontier", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 2: Constrained Optimization: Best BPB vs CE Ceiling
    # -------------------------------------------------------------
    ax2 = axs[0, 1]
    
    taus = [c["ce_ceiling"] for c in ce_constrained if c["best_bpb"] is not None]
    best_bpbs = [c["best_bpb"] for c in ce_constrained if c["best_bpb"] is not None]
    
    ax2.step(taus, best_bpbs, where="post", color="#1b9e77", linewidth=2.5, label="Optimal BPB Frontier")
    
    for c in ce_constrained:
        if c["best_bpb"] is not None:
            col = tok_colors[c["tokenizer"]]
            ax2.scatter(c["ce_ceiling"], c["best_bpb"], color=col, s=140, edgecolors="black", zorder=4)
            lbl = f"{c['tokenizer'].split('-')[0]} ({c['vocab']}-{c['tier'].split()[0][0]})"
            ax2.annotate(lbl, (c["ce_ceiling"], c["best_bpb"]),
                         xytext=(-15, 8), textcoords="offset points", fontsize=8.5, fontweight="bold",
                         color=col)
                         
    ax2.set_xlabel("Per-Token CE Ceiling tau_CE (nats)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Minimum Achievable BPB", fontsize=12, fontweight="bold")
    ax2.set_title("B: Constrained Decision: min(BPB) s.t. CE <= tau_CE", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 3: Parameter Efficiency: BPB vs Total Model Parameters
    # -------------------------------------------------------------
    ax3 = axs[1, 0]
    
    for tok in tok_colors:
        sub = [d for d in dataset if d["tokenizer"] == tok]
        params_m = [d["total_params"] / 1e6 for d in sub]
        bpb_vals = [d["bpb"] for d in sub]
        ax3.scatter(params_m, bpb_vals, color=tok_colors[tok], label=tok.split("-")[0],
                    marker="o", s=90, edgecolors="black", alpha=0.75)
        
    for d in dataset:
        if d["vocab_label"] == "64K" and d["tier"] == "Small (4L-128d)" and "Caliper" in d["tokenizer"]:
            ax3.annotate(f"Caliper 64K-Small\n(17.6M, BPB={d['bpb']:.2f})", (d["total_params"]/1e6, d["bpb"]),
                         xytext=(10, -15), textcoords="offset points", fontsize=8.5, fontweight="bold",
                         color="#7570b3", arrowprops=dict(arrowstyle="->", color="#7570b3"))
        elif d["vocab_label"] == "16K" and d["tier"] == "Large (8L-512d)" and "Caliper" in d["tokenizer"]:
            ax3.annotate(f"Caliper 16K-Large\n(42.0M, BPB={d['bpb']:.2f})", (d["total_params"]/1e6, d["bpb"]),
                         xytext=(-80, 15), textcoords="offset points", fontsize=8.5, fontweight="bold",
                         color="#7570b3", arrowprops=dict(arrowstyle="->", color="#7570b3"))

    ax3.set_xlabel("Total Model Parameters (Millions)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("True LM BPB -> min", fontsize=12, fontweight="bold")
    ax3.set_title("C: Parameter Efficiency & Tokenizer-Model Co-Design", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 4: Tokenizer Dominance Regime Heatmap
    # -------------------------------------------------------------
    ax4 = axs[1, 1]
    
    grid_matrix = np.zeros((3, 3))
    text_labels = []
    
    tier_order = ["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"]
    vocab_order = ["16K", "32K", "64K"]
    
    for r, tier in enumerate(tier_order):
        row_labels = []
        for c, v_label in enumerate(vocab_order):
            sub = [d for d in dataset if d["tier"] == tier and d["vocab_label"] == v_label]
            best_bpb = min(sub, key=lambda x: x["bpb"])
            best_ce = min(sub, key=lambda x: x["ce"])
            
            tok_map = {"SentencePiece-Unigram": 0, "Boundary-BPE": 1, "Caliper-SuperBPE (Config B)": 2}
            grid_matrix[r, c] = tok_map[best_bpb["tokenizer"]]
            
            lbl = f"BPB: {best_bpb['tokenizer'].split('-')[0]}\nCE: {best_ce['tokenizer'].split('-')[0]}"
            row_labels.append(lbl)
        text_labels.append(row_labels)
        
    cmap = matplotlib.colors.ListedColormap(["#2b5c8f", "#d95f02", "#7570b3"])
    cax = ax4.imshow(grid_matrix, cmap=cmap, aspect="auto", alpha=0.4, vmin=0, vmax=2)
    
    for r in range(3):
        for c in range(3):
            ax4.text(c, r, text_labels[r][c], ha="center", va="center", fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
                     
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(vocab_order, fontsize=11, fontweight="bold")
    ax4.set_yticks(range(3))
    ax4.set_yticklabels(["Small", "Medium", "Large"], fontsize=11, fontweight="bold")
    ax4.set_xlabel("Vocabulary Capacity", fontsize=12, fontweight="bold")
    ax4.set_ylabel("LM Architecture Tier", fontsize=12, fontweight="bold")
    ax4.set_title("D: Dominance Regime Matrix (argmin BPB & argmin CE)", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Phase 15 Pareto Frontier figure to {out_path}", flush=True)

def main():
    print("===============================================================================================================================================================================")
    print("PHASE FIFTEEN: COMPUTE-NORMALIZED PARETO FRONTIER & MULTI-OBJECTIVE DECISION ANALYSIS")
    print("Evaluating 27 Total Conditions (3 Vocab Scales x 3 LM Tiers x 3 Tokenizers)")
    print("===============================================================================================================================================================================\n")
    
    dataset = load_all_records()
    print(f"Loaded {len(dataset)} distinct experimental configurations.\n")
    
    # 1. Compute Global Pareto Frontier
    pareto_points = compute_pareto_frontier(dataset)
    
    print("===============================================================================================================================================================================")
    print("GLOBAL NON-DOMINATED PARETO FRONTIER SET in (BPB, CE) OBJECTIVE SPACE")
    print("===============================================================================================================================================================================")
    print(f"{'Tokenizer':<30} | {'Tier':<18} | {'Vocab':<6} | {'True BPB':<10} | {'Token CE (nats)':<16} | {'B/Tok':<8} | {'Params':<10} | {'Pareto Role'}")
    print("-" * 140)
    for p in pareto_points:
        role = ""
        if p["bpb"] == min(d["bpb"] for d in dataset):
            role += "[Global argmin BPB] "
        if p["ce"] == min(d["ce"] for d in dataset):
            role += "[Global argmin CE] "
        if not role:
            role = "[Intermediate Tradeoff]"
        print(f"{p['tokenizer']:<30} | {p['tier']:<18} | {p['vocab_label']:<6} | {p['bpb']:<10.3f} | {p['ce']:<16.3f} | {p['bytes_per_token']:<8.2f} | {p['total_params']:<10} | {role}")
    print("-" * 140 + "\n")
    
    # Tier-by-Tier Pareto Sets
    print("===============================================================================================================================================================================")
    print("TIER-SPECIFIC PARETO NON-DOMINATED FRONTIERS")
    print("===============================================================================================================================================================================")
    for tier in ["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"]:
        tier_sub = [d for d in dataset if d["tier"] == tier]
        tier_pareto = compute_pareto_frontier(tier_sub)
        print(f"--- [Tier: {tier}] ---")
        for p in tier_pareto:
            print(f"  {p['tokenizer']:<28} | Vocab: {p['vocab_label']:<4} | BPB: {p['bpb']:<6.3f} | CE: {p['ce']:<6.3f} | B/Tok: {p['bytes_per_token']:<5.2f}")
    print("-" * 140 + "\n")
    
    # 2. Constrained Optimization
    ce_constrained, bpb_constrained, param_constrained = run_constrained_decision_analysis(dataset)
    
    print("===============================================================================================================================================================================")
    print("CONSTRAINED DECISION ANALYSIS: min(BPB) SUBJECT TO TOKEN CE CEILING (CE <= tau_CE)")
    print("===============================================================================================================================================================================")
    print(f"{'CE Ceiling tau_CE':<18} | {'Best Achievable BPB':<20} | {'Optimal Tokenizer':<30} | {'LM Tier':<18} | {'Vocab':<6} | {'CE at Opt':<10}")
    print("-" * 120)
    for c in ce_constrained:
        if c["best_bpb"] is not None:
            print(f"{c['ce_ceiling']:<18.1f} | {c['best_bpb']:<20.3f} | {c['tokenizer']:<30} | {c['tier']:<18} | {c['vocab']:<6} | {c['best_ce']:<10.3f}")
        else:
            print(f"{c['ce_ceiling']:<18.1f} | {'INFEASIBLE':<20} | {'-':<30} | {'-':<18} | {'-':<6} | {'-':<10}")
    print("-" * 120 + "\n")
    
    print("===============================================================================================================================================================================")
    print("CONSTRAINED DECISION ANALYSIS: min(CE) SUBJECT TO BPB CEILING (BPB <= tau_BPB)")
    print("===============================================================================================================================================================================")
    print(f"{'BPB Ceiling tau_BPB':<20} | {'Best Achievable CE':<20} | {'Optimal Tokenizer':<30} | {'LM Tier':<18} | {'Vocab':<6} | {'BPB at Opt':<10}")
    print("-" * 120)
    for b in bpb_constrained:
        if b["best_ce"] is not None:
            print(f"{b['bpb_ceiling']:<20.1f} | {b['best_ce']:<20.3f} | {b['tokenizer']:<30} | {b['tier']:<18} | {b['vocab']:<6} | {b['best_bpb']:<10.3f}")
        else:
            print(f"{b['bpb_ceiling']:<20.1f} | {'INFEASIBLE':<20} | {'-':<30} | {'-':<18} | {'-':<6} | {'-':<10}")
    print("-" * 120 + "\n")

    print("===============================================================================================================================================================================")
    print("CO-DESIGN DECISION ANALYSIS: min(BPB) SUBJECT TO PARAMETER FOOTPRINT (Params <= tau_P)")
    print("===============================================================================================================================================================================")
    print(f"{'Param Ceiling (M)':<20} | {'Best Achievable BPB':<20} | {'Optimal Tokenizer':<30} | {'LM Tier':<18} | {'Vocab':<6} | {'Actual Params (M)':<18}")
    print("-" * 120)
    for p in param_constrained:
        print(f"{p['param_ceiling_m']:<20.1f} | {p['best_bpb']:<20.3f} | {p['tokenizer']:<30} | {p['tier']:<18} | {p['vocab']:<6} | {p['total_params']/1e6:<18.2f}")
    print("-" * 120 + "\n")

    benchmarks_dir = r"c:\Users\shaik\Research\Tokenizer\benchmarks"
    fig_path = os.path.join(benchmarks_dir, "phase_fifteen_pareto_frontier.png")
    json_path = os.path.join(benchmarks_dir, "phase_fifteen_pareto_records.json")
    
    plot_pareto_frontier(dataset, pareto_points, ce_constrained, fig_path)
    
    output_ledger = {
        "all_configurations": dataset,
        "pareto_optimal_set": pareto_points,
        "ce_constrained_decisions": ce_constrained,
        "bpb_constrained_decisions": bpb_constrained,
        "param_constrained_decisions": param_constrained
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_ledger, f, indent=2)
    print(f"[Ledger] Saved Phase 15 Pareto ledger to {json_path}\n", flush=True)

if __name__ == "__main__":
    main()
