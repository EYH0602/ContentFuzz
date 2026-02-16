#!/usr/bin/env python3
"""Summarize NLI results from nli.txt into per-dataset tables."""

import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser(
        description="Summarize NLI results into per-dataset tables."
    )
    parser.add_argument("logfile", help="Path to the NLI log file")
    args = parser.parse_args()

    with open(args.logfile) as f:
        text = f.read()

    # Extract the JSON array from the SUMMARY section
    match = re.search(r"SUMMARY\n=+\n(\[.*\])", text, re.DOTALL)
    records = json.loads(match.group(1))

    # Aggregate by dataset (weighted average by n_pairs)
    datasets = {}
    for r in records:
        ds = r["dataset"]
        if ds not in datasets:
            datasets[ds] = {"lang": r["lang"], "forward": [], "backward": []}
        datasets[ds]["forward"].append(r["nli"]["forward"])
        datasets[ds]["backward"].append(r["nli"]["backward"])

    def weighted_avg(entries):
        total_n = sum(e["n_pairs"] for e in entries)
        ent = sum(e["entailment"] * e["n_pairs"] for e in entries) / total_n
        neu = sum(e["neutral"] * e["n_pairs"] for e in entries) / total_n
        con = sum(e["contradiction"] * e["n_pairs"] for e in entries) / total_n
        return total_n, ent, neu, con

    # Desired row order
    row_order = [
        ("sem16", "Sem16", "EN"),
        ("vast", "VAST", "EN"),
        ("c-stance-a", "C-STANCE-A", "ZH"),
    ]

    for direction, label in [
        ("forward", "Original → Rewrite"),
        ("backward", "Rewrite → Original"),
    ]:
        print(f"**Table: NLI Analysis ({label})**\n")
        print(
            f"| Dataset    | Lang | N     | Entail (%) | Neutral (%) | Contradict (%) |"
        )
        print(
            f"|------------|------|-------|------------|-------------|----------------|"
        )

        all_entries = []
        for ds_key, ds_name, lang in row_order:
            entries = datasets[ds_key][direction]
            all_entries.extend(entries)
            n, ent, neu, con = weighted_avg(entries)
            print(
                f"| {ds_name:<10} | {lang}   | {n:<5} | {ent:>10.2f} | {neu:>11.2f} | {con:>14.2f} |"
            )

        # All row
        n, ent, neu, con = weighted_avg(all_entries)
        print(
            f"| {'**All**':<10} |      | {n:<5} | {ent:>10.2f} | {neu:>11.2f} | {con:>14.2f} |"
        )
        print()


if __name__ == "__main__":
    main()
