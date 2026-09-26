#!/bin/bash
# SageMaker entry: pull eval/(pool) from S3, run ucfm, write logs/checkpoints back.
# Credentials come from ucfm-sagemaker-role. Set UCFM_S3_BUCKET (and UCFM_TASK) on the job.
set -euo pipefail

: "${UCFM_S3_BUCKET:?set UCFM_S3_BUCKET on the SageMaker training job}"
TASK="${UCFM_TASK:-baseline}"

echo "task=${TASK} bucket=${UCFM_S3_BUCKET} cuda=$(python -c 'import torch; print(torch.cuda.is_available())')"

ucfm s3-check
ucfm sync-eval

if [[ "${TASK}" == "train" ]]; then
  ucfm sync-pool || true
  ucfm train
else
  ucfm baseline --force
fi

echo "done ${TASK}"
