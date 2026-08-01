ARG BASE_IMAGE=docker.io/pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="VGGTBEV Method I training"
LABEL org.opencontainers.image.description="NCCL/DDP Method I pipeline with four HM3D smoke scenes"
LABEL org.opencontainers.image.version="0.7.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/vggtbev/src:/opt/vggt

COPY vggt_omega/ /opt/vggt/vggt_omega/
COPY model.pt /opt/vggt/checkpoints/VGGT-Omega-1B-512/model.pt
COPY method1/src/ /opt/vggtbev/src/
COPY method1/configs/ /opt/vggtbev/configs/
COPY testdata/ /opt/vggtbev/testdata/
COPY --chmod=755 method1/docker/entrypoint.sh /usr/local/bin/vggtbev-method1

RUN mkdir -p /opt/vggtbev/manifests /workspace/runs \
    && python -m vggt_bev_method1.cli_freeze_split \
        --config /opt/vggtbev/configs/docker_4scene_single.toml \
    && python -c "from vggt_omega.models import VGGTOmega; from vggt_bev_method1.models import Method1System"

WORKDIR /workspace
ENTRYPOINT ["vggtbev-method1"]
CMD ["validate"]
