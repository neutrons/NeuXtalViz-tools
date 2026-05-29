#!/bin/sh

#User command script which is what will execute the desired user program.
#Should be modified to run the desired program and remove this tail command.

cd $DEFAULT_DIR
pixi run -m /opt/NeuXtalViz/pixi.toml python /opt/NeuXtalViz/src/NeuXtalViz.py
