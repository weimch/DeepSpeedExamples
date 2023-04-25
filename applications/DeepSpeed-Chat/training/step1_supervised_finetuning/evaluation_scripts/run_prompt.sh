#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
python3.8 prompt_eval.py \
    --model_name_or_path_baseline THUDM/chatglm-6b-int4 \
    --model_name_or_path_finetune /data/home/minchangwei/DeepSpeedExamples/applications/DeepSpeed-Chat/output/01-mine-actor-model/pytorch_model.bin \
