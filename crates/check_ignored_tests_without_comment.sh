#!/usr/bin/env bash

grep -n -r '#\[ignore\]' $1/src | grep -v '\/\/'
if [ "$?" -ne "1" ]; then
    echo "there are ignored test cases without accompanying comment"
    exit 1
fi
