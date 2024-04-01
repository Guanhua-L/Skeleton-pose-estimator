#!/bin/bash

if [ ! -d "./skeleton" ]; then
  echo "skeleton文件夹不存在"
  exit 1
fi

cd ./skeleton

for file in *.csv; do
  if [ -e "$file" ]; then
    echo "doing: $file"
    python3 ../visualize_skeleton.py "$file"
  else
    echo "no file $file"
  fi
done

cd ..
