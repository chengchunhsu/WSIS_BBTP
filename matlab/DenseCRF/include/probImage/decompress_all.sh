#!/bin/bash
EXE=build/decompress

if [ $# -lt 2 ]; then
	echo "Usage: $0 compressed_dir unary_dir"
	exit 1
fi

for i in $1/*.c_unary; do
	PATH=$i
	NAME=${i/$1\//}
	TARGET=${NAME/c_unary/unary}
	TARGET_PATH=$2/$TARGET
	$EXE $PATH $TARGET_PATH
done
