# SPIRAL Placer - James Bennett

## Run
```bash
uv run evaluate submissions/hybrid_analytical_placer.py --all
```

<<<<<<< HEAD
### Run Your First Example

```bash
# Run the greedy row placer on ibm01
uv run evaluate submissions/examples/greedy_row_placer.py

# Run on all 17 IBM benchmarks
uv run evaluate submissions/examples/greedy_row_placer.py --all

# Run on NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla)
uv run evaluate submissions/examples/greedy_row_placer.py --ng45

# Visualize the result
uv run evaluate submissions/examples/greedy_row_placer.py --vis
uv run evaluate submissions/examples/greedy_row_placer.py --all --vis
```

Running on all benchmarks produces a summary like:
```
Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
   ibm01    2.0463    1.3166    0.9976    -55.4%     -105.1%         0
   ibm02    2.0431    1.9072    1.8370     -7.1%      -11.2%         0
   ...
     AVG    2.2109    2.1251    1.4578     -4.0%      -51.7%         0
```

The greedy placer achieves zero overlaps but makes no attempt to optimize wirelength or connectivity — your job is to do better! See [`SETUP.md`](SETUP.md) for the full API reference and [`submissions/examples/`](submissions/examples/) for working examples.

## 🎯 IBM Benchmark Suite (ICCAD04)

We evaluate on the complete ICCAD04 IBM benchmark suite:

| Benchmark | Macros | Nets | Canvas (μm) | Area Util. | SA Baseline | RePlAce Baseline |
|-----------|--------|------|-------------|------------|-------------|------------------|
| **ibm01** | 246 | 7,269 | 22.9×23.0 | 42.8% | 1.3166 | **0.9976** ⭐ |
| **ibm02** | 254 | 7,538 | 23.2×23.5 | 43.1% | 1.9072 | **1.8370** ⭐ |
| **ibm03** | 269 | 8,045 | 24.1×24.3 | 44.2% | 1.7401 | **1.3222** ⭐ |
| **ibm04** | 285 | 8,654 | 24.8×25.1 | 44.8% | 1.5037 | **1.3024** ⭐ |
| **ibm06** | 318 | 9,745 | 26.1×26.5 | 46.1% | 2.5057 | **1.6187** ⭐ |
| **ibm07** | 335 | 10,328 | 26.8×27.2 | 46.8% | 2.0229 | **1.4633** ⭐ |
| **ibm08** | 352 | 10,901 | 27.5×27.9 | 47.4% | 1.9239 | **1.4285** ⭐ |
| **ibm09** | 369 | 11,463 | 28.1×28.5 | 48.0% | 1.3875 | **1.1194** ⭐ |
| **ibm10** | 387 | 12,018 | 28.8×29.2 | 48.6% | 2.1108 | **1.5009** ⭐ |
| **ibm11** | 405 | 12,568 | 29.4×29.8 | 49.2% | 1.7111 | **1.1774** ⭐ |
| **ibm12** | 423 | 13,111 | 30.1×30.5 | 49.8% | 2.8261 | **1.7261** ⭐ |
| **ibm13** | 441 | 13,647 | 30.7×31.1 | 50.4% | 1.9141 | **1.3355** ⭐ |
| **ibm14** | 460 | 14,178 | 31.4×31.8 | 51.0% | 2.2750 | **1.5436** ⭐ |
| **ibm15** | 479 | 14,704 | 32.0×32.4 | 51.6% | 2.3000 | **1.5159** ⭐ |
| **ibm16** | 498 | 15,225 | 32.7×33.1 | 52.2% | 2.2337 | **1.4780** ⭐ |
| **ibm17** | 517 | 15,741 | 33.3×33.7 | 52.8% | 3.6726 | **1.6446** ⭐ |
| **ibm18** | 537 | 16,253 | 34.0×34.4 | 53.4% | 2.7755 | **1.7722** ⭐ |

Each benchmark includes:
- Hard macros (you place these)
- Soft macros (you can also place these)
- Nets connecting all components
- Initial placement (hand-crafted, serves as reference)

**Baseline Analysis:**
- RePlAce (⭐) consistently outperforms SA across all benchmarks
- RePlAce achieves 15-55% lower proxy cost than SA
- **To qualify for the Grand Prize, your placement must also produce better WNS, TNS, and Area than both baselines when evaluated through the OpenROAD flow on NG45 designs**
- Both baselines achieve zero overlaps (enforced as hard constraint)

## 💡 Why This Is Hard

Despite "only" 246-537 macros, this problem is extremely challenging:

1. **Massive search space**: ~10^800 possible placements (even with constraints)
2. **Conflicting objectives**: Wirelength wants clustering, density wants spreading, congestion wants routing space
3. **Non-convex landscape**: Millions of local minima, discontinuities, plateaus
4. **Long-range dependencies**: Moving one macro affects costs globally through thousands of nets
5. **Hard constraints**: No overlaps between heterogeneous sizes (33× size variation)
6. **Tight packing**: 43-53% area utilization leaves little slack
7. **Runtime matters**: Must be fast enough to be practical (< 5 minutes ideal)

Classical methods (SA, RePlAce) have been refined for decades but still have room for improvement!

## 📖 Documentation

- **Setup & API Reference**: [`SETUP.md`](SETUP.md) - Infrastructure details, benchmark format, cost computation, testing
- **Example Submissions**: [`submissions/examples/`](submissions/examples/) - Working placer examples

## 📚 References

- **TILOS MacroPlacement**: [GitHub Repository](https://github.com/TILOS-AI-Institute/MacroPlacement)
  - Source of evaluation infrastructure
  - ICCAD04 benchmarks
  - SA and RePlAce baseline implementations

- **ICCAD04 Benchmarks**: Classic macro placement benchmark suite used in academic research

## 🏅 Leaderboard

Submissions are ranked by **average proxy cost** across all 17 IBM benchmarks (lower is better). Zero overlaps required on all benchmarks. Scores are unverified until confirmed by judges.

| Rank | Team | Avg Proxy Cost | Best | Worst | Overlaps | Runtime | Verified | Notes |
|------|------|---------------|------|-------|----------|---------|----------|-------|
| 1 | "vmallela" | **1.0109** | 0.7644 | 1.2921 | 0 | 15.5h total | :white_check_mark: | Verified 1.0109 (self-reported 1.1) |
| 2 | "Cezar" | **1.037** | — | — | 0 | 55min/bench | | Resubmitted 5/3. Verification blocked on missing `steps/analytical.py` import. Previous variant verified 1.2224. |
| 3 | "KLA MACH" | **1.2121** | 0.8527 | 1.6532 | 0 | 2h15min total | :white_check_mark: | Verified 1.2121 (self-reported 1.2355). Consolidates UTDA / Chuanqi Chen / KLA MACH submissions (one algorithm per team). |
| 4 | "Hoop Dreams" | **1.2207** | 0.8972 | 1.5072 | 0 | 5h total | :white_check_mark: | Verified 1.2207 (self-reported 1.2206). Built and ran from team-provided `Dockerfile`. |
| 5 | "Shoom" | **1.2353** | — | — | 0 | 42min/bench | | Resubmitted 5/1 (was 1.3381). |
| 6 | "ArzunPD" | **1.2478** | — | — | 0 | 55min/bench | | Resubmitted 5/1 (was HyperPlace, verified 1.4421). |
| 7 | "RoRa" | **1.2788** | 0.9577 | 1.6222 | 0 | 2.6h total | :white_check_mark: | Verified 1.2788 (self-reported 1.2723). Resubmitted 5/1. |
| 8 | "William Zhang" | **1.2767** | — | — | 0 | 259s/bench | | Resubmitted 5/2 (was "Convex Optimization", verified 1.4556). Blocked on missing `casadi` module. |
| 9 | "MTK" | **1.2818** | 0.9073 | 1.6529 | 0 | 37s/bench (GPU) | :white_check_mark: | Verified 1.2818 (self-reported 1.317). |
| 10 | "Electric Beatle" | **1.3253** | — | — | 0 | 2000s/bench (GPU) | | Resubmitted 4/30 (was verified 1.3913). |
| 11 | "UToronto Analytical" | **1.3323** | 0.9371 | 1.6545 | 0 | 24min total | :white_check_mark: | Verified 1.3323 (self-reported 1.3325). |
| 12 | "V5" | **1.3382** | — | — | 0 | 850s/bench | | New 4/23. |
| 13 | "Archgen" | **1.3479** | — | — | 0 | 2404s total | | New 4/24. |
| 14 | "Varun's Parallel Worlds" | **1.4017** | 1.0362 | 1.7298 | 0 | 27s/bench | :white_check_mark: | |
| 15 | "UT Austin - AS" | **1.4076** | — | — | 0 | 17s/bench | | |
| 16 | "ByteDancer" | **1.4151** | 1.0236 | 1.7792 | 0 | 38min/bench | :white_check_mark: | |
| 17 | "TAISPlAce" | **1.4321** | — | — | 0 | 28min/bench | | |
| 18 | "Two-IIITK-Kids" | **1.436** | — | — | 0 | 38min/bench | | New 5/2, resubmitted 5/4. |
| 19 | "Pragnay" | **1.4427** | — | — | 0 | 632s/bench | | Blocked on `compute_proxy_cost(..., plc=None)` in fallback path. |
| 20 | "No Man's Sky" | **1.4445** | — | — | 0 | 8.8min/bench | | New 5/4. Repo not accessible. |
| 21 | "another Waterloo kid" | **1.4568** | — | — | 0 | 118s/bench | | Blocked on Modal cloud dispatch — can't run air-gapped. |
| — | RePlAce (baseline) | **1.4578** | 0.9976 | 1.8370 | 0 | — | :white_check_mark: | |
| 22 | "W3 Solutions" | **1.4824** | — | — | 0 | 90s/bench | | Runtime exceeds 1h/bench cap. |
| 23 | "Jiangban Ya" | **1.4943** | 1.0891 | 1.8099 | 0 | 49s/bench | :white_check_mark: | |
| 24 | "UTAUSTIN-CT" | **1.5062** | 1.1363 | 1.7941 | 0 | 6s/bench | :white_check_mark: | |
| 25 | "oracleX" | **1.5130** | 1.1340 | 1.7937 | 0 | 11s/bench | :white_check_mark: | |
| 26 | "SEVmakers" | **1.5200** | — | — | 0 | 200s/bench | | Private repo — pending judge access. |
| 27 | "CA" | **1.5247** | 1.2226 | 1.7945 | 0 | 2s/bench | :white_check_mark: | Verified 1.5247 (self-reported 1.5238). |
| 28 | "#5 ubc cpen student" | **1.5337** | 1.1411 | 1.8084 | 0 | 13s/bench | :white_check_mark: | |
| 29 | Will Seed (Partcl) | **1.5338** | 1.1625 | 1.7965 | 0 | 35s total | :white_check_mark: | |
| 30 | "RUDY Can't Fail" | **1.5397** | 1.1927 | 1.8881 | 0 | 6min total | :white_check_mark: | Verified 1.5397 (self-reported 1.3605). |
| 31 | "UT Austin - RH" | **1.6037** | — | — | 0 | 4.5s/bench | | |
| 32 | "UT Austin - CT" | **1.8706** | — | — | 0 | 187s/bench | | |
| 33 | "AS" | **1.9121** | 1.4614 | 2.3508 | 0 | 0.16s total | :white_check_mark: | |
| 34 | "Adi's Team" | **2.0025** | — | — | 0 | 3726s/bench | | Blocked on `compute_proxy_cost(skip_congestion=True)` kwarg. |
| 35 | "Sharc #1" | **2.0433** | 1.5143 | 2.4336 | 0 | 96s/bench | :white_check_mark: | |
| — | SA (baseline) | 2.1251 | 1.3166 | 3.6726 | 0 | — | :white_check_mark: | |
| — | Greedy Row (demo) | 2.2109 | 1.6728 | 2.7696 | 0 | 0.3s total | :white_check_mark: | |
| — | "Binghamton" | pending | — | — | — | — | | |
| — | "MacroBio" | pending | — | — | — | — | | |
| DQ | "Mike Gao" | self-reported 1.3255 | — | — | 1939 | 16min/bench | | 1939 overlaps across 17 benchmarks. |
| DQ | "BakaBobo" | self-reported 1.4044 | — | — | — | 282s/bench | | Missing import — code won't run. |

*Submit your results via the [Submission Link](https://forms.gle/YDRtYV5Vq68SZgKW9)!*

## 🤔 FAQ

**Q: What benchmarks are used?**
A: Tier 1 (proxy cost) uses 17 IBM ICCAD04 benchmarks — the standard academic suite with well-established baselines. Tier 2 (OpenROAD flow) uses NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla) plus 1-2 hidden designs. You can evaluate on both with `--all` (IBM) and `--ng45` (NG45).

**Q: What if I beat one baseline but not the other?**
A: You must beat BOTH SA and RePlAce baselines on WNS, TNS, and Area to qualify for the Grand Prize. You can still win the Proxy or Innovation prizes regardless.

**Q: Are there hidden test cases?**
A: All 17 IBM benchmarks for proxy cost ranking are public. The 4 NG45 designs are also public. For the OpenROAD flow evaluation (Tier 2), we will additionally test on 1-2 hidden NG45 designs to ensure generalization.

**Q: What counts as "beating" the baseline?**
A: For proxy cost (Tier 1), your aggregate score across all IBM benchmarks must be lower than the baselines. For the Grand Prize (Tier 2), your OpenROAD results for WNS, TNS, and Area must surpass both SA and RePlAce baselines on NG45 designs.

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/partcleda/partcl-macro-place-challenge/issues)
- **Email**: contact@partcl.com

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE.md](LICENSE.md) for details.

## Competition Updates

The organizers may update or clarify rules, evaluation details, timelines, prizes, or infrastructure as needed to ensure fairness, technical accuracy, and smooth operation of the competition. Any updates will be communicated through official channels and will apply going forward.

Participation in the competition constitutes acceptance of the current rules and any subsequent updates. The organizers’ decisions regarding scoring, eligibility, and interpretation of these rules are final.

Submissions & contact information may be shared with sponsors.
=======
## Dependencies
torch, numpy, scipy, numba, matplotlib, tqdm, absl-py
(all included in the competition eval Docker image)
>>>>>>> c7cc296 (Update README.md)
