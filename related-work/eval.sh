#!/usr/bin/env bash

for ds in SWaT  HAI BATADAL; do
    for ids in Invariant SIMPLE Seq2SeqNN TABOR PASAD Geco; do
        echo -e "\n$ds $ids"
        ipal-evaluate \
            --attacks ../datasets/$ds/attacks.json \
            --output $ds/$ids.json \
            $ds/$ids.state.gz
    done
done
