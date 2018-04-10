#!/bin/bash

for var in "$@"; do
	python /remote/64bin/auto_astrom/esa_gaia.py "$var"
	
done
