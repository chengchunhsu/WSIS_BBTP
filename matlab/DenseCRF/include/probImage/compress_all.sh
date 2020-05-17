#!/bin/bash
EXE=build/compress

if [ $# -lt 2 ]; then
	echo "Usage: $0 unary_dir compressed_dir"
	exit 1
fi

for i in $1/*.unary; do
	PATH=$i
	NAME=${i/$1\//}
	TARGET=${NAME/unary/c_unary}
	TARGET_PATH=$2/$TARGET
	$EXE $PATH $TARGET_PATH
done
