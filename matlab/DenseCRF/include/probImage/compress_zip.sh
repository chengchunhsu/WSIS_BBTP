#!/bin/bash
DIR=$(mktemp -d)
bash compress_all.sh $1 $DIR
zip -j $2 $DIR/*
rm -rf $DIR
