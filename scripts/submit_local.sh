#!/bin/bash

#Export Parameters
export M=50000000
export T=200000
export CPU=14

./run_local.sh && ./plot_local.sh
