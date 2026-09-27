"""Appendix C: arithmetic and inference checks on Tables C.1 / C.2 (values transcribed from the paper).

Run from the repo root:  .venv/bin/python audit/scripts/appendix_c_checks.py
"""
import itertools
import numpy as np
from scipy.stats import spearmanr

S = ["zero", "structured", "cot", "few"]
C1 = {  # task -> model -> [zero, structured, cot, few]
    "dx":      {"GPT": [.676, .665, .710, .732], "DeepSeek": [.664, .633, .633, .652], "Mistral": [.578, .587, .699, .611]},
    "summ":    {"GPT": [.320, .411, .269, .296], "DeepSeek": [.215, .323, .175, .209], "Mistral": [.301, .314, .246, .209]},
    "retr":    {"GPT": [.854, .764, .785, .785], "DeepSeek": [.803, .727, .678, .761], "Mistral": [.844, .785, .743, .843]},
    "imaging": {"GPT": [.525, .476, .485, .544], "DeepSeek": [.515, .484, .464, .537], "Mistral": [.399, .439, .403, .465]},
}
print("C.1 cells:", sum(len(v) * 4 for v in C1.values()), "(paper text says 60)")
for t, ms in C1.items():
    wins = {m: S[int(np.argmax(v))] for m, v in ms.items()}
    mean = np.mean(list(ms.values()), axis=0)
    print(f"  {t:8s} per-model winner {wins} | winner of the 3-model mean: {S[int(np.argmax(mean))]} "
          f"({', '.join(f'{s}={x:.3f}' for s, x in zip(S, mean))})")

D = {"Kimi": [.025, -.004, -.074, -.032], "GPT": [-.010, -.003, -.144, -.012], "Opus": [.003, .003, -.087, -.020], "GLM": [-.021, .027, -.058, -.041]}
tasks = ["dx", "summ", "retr", "img"]
print("C.2 mean|Δ|:", {m: round(float(np.mean(np.abs(v))), 4) for m, v in D.items()})
others = np.mean([D[m] for m in ("GPT", "Opus", "GLM")], axis=0)
print("C.2 change in Kimi's margin over the other three (Kimi Δ - mean others Δ):",
      dict(zip(tasks, np.round(np.array(D["Kimi"]) - others, 3))))
# exact null distribution of Spearman rho for n=4
rhos = [spearmanr([0, 1, 2, 3], p).correlation for p in itertools.permutations(range(4))]
print("n=4 Spearman: possible values", sorted({round(r, 1) for r in rhos}),
      f"| P(rho >= 0.8 | random order) = {np.mean([r >= 0.8 - 1e-9 for r in rhos]):.3f}")
