import re
import pandas as pd
from pathlib import Path

START_FINAL_RE = re.compile(r"\[(ibm\d+)\]\s+(start|final)\s+proxy=([\d.]+)")
DONE_RE = re.compile(
    r"^(SA swap|SA soft swap|Soft spread|SA soft displace|SA displace|Channel relocate|Soft channel relocate).*done:.*?(?:best (?:real_)?proxy|best proxy)=([\d.]+)",
    re.I,
)
NESTEROV_RE = re.compile(r"legalized checkpoint from step \d+ has proxy cost ([\d.]+)")
POLISH_RE = re.compile(r"\[polish\] done: proxy ([\d.]+) → ([\d.]+)")

def parse_log(path):
    rows = []
    current_bench = None
    current_proxy = None
    nesterov_candidates = []

    text = Path(path).read_text(errors="ignore").splitlines()

    for line in text:
        m = START_FINAL_RE.search(line)
        if m:
            bench, tag, proxy = m.group(1), m.group(2), float(m.group(3))
            current_bench = bench
            if tag == "start":
                current_proxy = proxy
            elif tag == "final":
                rows.append([bench, "final reported", current_proxy, proxy, proxy - current_proxy])
                current_proxy = proxy
            continue

        m = NESTEROV_RE.search(line)
        if m:
            nesterov_candidates.append(float(m.group(1)))
            continue

        if "SA swap budget" in line and nesterov_candidates:
            after = min(nesterov_candidates)
            rows.append([current_bench, "nesterov/legalized", current_proxy, after, after - current_proxy])
            current_proxy = after
            nesterov_candidates = []
            continue

        m = DONE_RE.search(line)
        if m:
            phase = m.group(1).lower()
            after = float(m.group(2))
            rows.append([current_bench, phase, current_proxy, after, after - current_proxy])
            current_proxy = after
            continue

        m = POLISH_RE.search(line)
        if m:
            before, after = float(m.group(1)), float(m.group(2))
            rows.append([current_bench, "polish", before, after, after - before])
            current_proxy = after
            continue

    df = pd.DataFrame(rows, columns=["bench", "phase", "proxy_before", "proxy_after", "delta"])
    df["helpful"] = (-df["delta"] / df["proxy_before"]) > 0.002
    return df

log_dfs = []
for log in ['./logs/ibm01.log', './logs/ibm02.log', './logs/ibm03.log',
            './logs/ibm04.log', './logs/ibm06.log', './logs/ibm07.log',
             './logs/ibm08.log', './logs/ibm09.log', './logs/ibm10.log',
             './logs/ibm11.log', './logs/ibm12.log', './logs/ibm13.log',
             './logs/ibm14.log', './logs/ibm15.log',  './logs/ibm16.log',
            './logs/ibm17.log', './logs/ibm18.log']:
    df = parse_log(log)
    log_dfs.append(df)

df = pd.concat(log_dfs)    
summary = (
    df.groupby("phase")
      .agg(
          mean_delta=("delta", "mean"),
          median_delta=("delta", "median"),
          helpful_rate=("helpful", "mean"),
          total_gain=("delta", "sum"),
      )
      .sort_values("total_gain")
)

print(summary)

