#!/bin/bash

"$HOME"/.local/bin/uv -n run train --multirun \
    data.latent_dim=2,3,4,8 \
    optim.lr=0.01,0.001,0.0001 \
    train.batch_size=32,64,128 \
    train.epochs=10,20,30 \
    > logs/out/multirun.log \
    2> logs/err/multirun.log &
