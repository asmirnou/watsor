#!/usr/bin/env bash

docker images -f "dangling=true" -q \
    | xargs -L1 -r docker inspect \
   	| jq --arg key "watsor.builder" -r '.[] | select(.Config.Labels[$key]) | [.Id, .Config.Labels[$key]] | @tsv' \
   	| xargs -L1 -r docker tag
