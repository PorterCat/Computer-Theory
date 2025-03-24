#!/bin/bash

find . -type f \( -name "*.o" -o -name "*.out" \) -exec rm -f {} \;