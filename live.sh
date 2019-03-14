#!/bin/bash

key=$1
echo "$key" >> /tmp/keys

python /src/tensorflow.py --test "$key"
