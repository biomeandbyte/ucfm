# SageMaker GPU training image (linux/amd64). Data stays in S3; this image is code + CUDA.
# Build: docker build --platform linux/amd64 -t ucfm-train:latest .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/ml/code

COPY pyproject.toml README.md config.yaml ./
COPY src ./src
COPY datasets ./datasets
COPY sagemaker/train.sh ./sagemaker/train.sh

# Keep the CUDA torch from the base image; do not let pip install a CPU wheel.
RUN pip install --no-cache-dir -e . --no-deps \
    && pip install --no-cache-dir \
        "boto3>=1.34" \
        "matplotlib>=3.8" \
        "numpy>=1.24" \
        "pillow>=10.0" \
        "pydantic>=2.6" \
        "pydantic-settings>=2.2" \
        "python-dotenv>=1.0" \
        "pyyaml>=6.0" \
        "scikit-learn>=1.3" \
        "transformers>=4.40" \
    && chmod +x sagemaker/train.sh

ENV PYTHONUNBUFFERED=1 \
    UCFM_TASK=baseline

ENTRYPOINT ["/bin/bash", "sagemaker/train.sh"]
