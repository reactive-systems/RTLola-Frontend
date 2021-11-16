#!/usr/bin/env bash

if ! [[ $(git diff --name-only HEAD HEAD~1 -- crates/$1/CHANGELOG.md) ]]; then
if [[ $(git diff --name-only HEAD HEAD~1 -- crates/$1/) ]]; then
    exit 1
fi
fi

exit 0