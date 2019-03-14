#!/bin/bash

key=$1
echo "$key" >> /tmp/keys

python /src/live.py "$key"
