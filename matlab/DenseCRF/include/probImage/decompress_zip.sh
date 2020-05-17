#!/bin/bash
DIR=$(mktemp -d)
unzip -j $1 -d $DIR/
bash decompress_all.sh $DIR $2
rm -rf $DIR
