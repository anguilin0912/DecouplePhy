#!/usr/bin/env bash
cd "$(dirname "$0")"

set -e  # abort on failure

ipal-iids \
    --log info  \
    --config ExpertInvariants.json \
    --train.state ../datasets/SWaT/SWaT_Dataset_Normal_v1.state.gz \
    --live.state ../datasets/SWaT/SWaT_Dataset_Attack_v0.state.gz \
    --output output.state.gz

ipal-minimize --log info --all output.state.gz

ipal-plot-alerts --min-width 120 --attacks ../datasets/SWaT/attacks.json --output output.pdf output.state.gz

ipal-evaluate --log info --attacks ../datasets/SWaT/attacks.json --output output.json output.state.gz
