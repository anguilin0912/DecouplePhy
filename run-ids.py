#!/usr/bin/env python3
import gzip
import json
import os
import sys

DATASETs = {
    "SWaT": {
        "train": "SWaT_Dataset_Normal_v1.state.gz",
        "test": ["SWaT_Dataset_Attack_v0.state.gz"],
        "skip": 80000,
        "gap": 1,
        "min-width": 600,
    },
    "WADI": {
        "train": "WADI_14days_new.state.gz",
        "test": ["WADI_attackdataLABLE.state.gz"],
        "skip": 0,
        "gap": 1,
        "min-width": 300,
    },
    "HAI": {
        "train": "train1.state.gz",
        "test": ["test1.state.gz", "test2.state.gz", "test3.state.gz", "test4.state.gz", "test5.state.gz",],
        "skip": 0,
        "gap": 1,
        "min-width": 600,
    },
    "BATADAL": {
        "train": "BATADAL_dataset03.state.gz",
        "test": ["BATADAL_dataset04.state.gz", "BATADAL_test_dataset.state.gz"],
        "skip": 0,
        "gap": 60,
        "min-width": 36000,
    },
    "TEP" : {
        "train": "train.state.gz",
        "test": ["DA1.state.gz", "DA2.state.gz", "SA1.state.gz", "SA2.state.gz", "SA3.state.gz"],
        "skip": 0,
        "gap": 1,
        "min-width": 1,
    },
}

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: ./run-ids.py [dataset name]")
        sys.exit(1)

    if sys.argv[1] not in DATASETs:
        print(f"Dataset unknown: {', '.join(DATASETs.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    ds = DATASETs[name]
    conf = f"./config/{name}.json"

    # Step 1: Training
    data = input("Retrain IDS model? Warning: may take long! (y/n):")
    if data.lower() in ["y", "yes"]:
        print(f"\n## 1 Training: {name}")
        train = f"./datasets/{name}/{ds['train']}"
        # 将 gzcat 改为 zcat
        err = os.system(f"zcat {train} | tail -n +{ds['skip']} | ipal-iids --log info --retrain --config {conf} --train.state -")
        if err != 0:
            print("Error while executing last command")
            sys.exit(1)
    else:
        print(f"\n## 1 Training: {name} SKIPPED")

    # Step 2: Live
    print(f"\n## 2 Live: {name}")
    for test in ds["test"]:
        live = f"./datasets/{name}/{test}"
        output = f"./output/{name}-{test}"
        err = os.system(f"ipal-iids --log info --config {conf} --live.state {live} --output {output}")
        if err != 0:
            print("Error while executing last command")
            sys.exit(1)

    # Step 3: Join
    print(f"\n## 3 Join: {name}")
    with gzip.open(f"./output/{name}.state.gz", "wt") as fout:
        for test in ds["test"]:
            with gzip.open(f"./output/{name}-{test}", "rt") as fin:
                fout.write(fin.read())
            os.remove(f"./output/{name}-{test}")

    # Step 4: Evaluate
    print(f"\n## 4 Evaluate: {name}")
    attacks = f"./datasets/{name}/attacks.json"
    output = f"./results/{name}.json"
    data = f"./output/{name}.state.gz"
    err = os.system(f"ipal-evaluate --attacks {attacks} --output {output} {data}")
    if err != 0:
        print("Error while executing last command")
        sys.exit(1)

    # Step 5: Plot
    print(f"\n## 5 Plot: {name}")
    err = os.system(f"ipal-plot-alerts --log info --attacks {attacks} --min-width 60 --mark-fp --draw-attack-id --max-gap {ds['gap']} --min-width {ds['min-width']} --title {name} --output ./results/{name}.pdf {data}")
    if err != 0:
        print("Error while executing last command")
        sys.exit(1)
