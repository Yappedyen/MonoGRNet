SHELL := /bin/bash

.PHONY: all-python3

all-python3:
	pip install runcython3
	makecython3++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

