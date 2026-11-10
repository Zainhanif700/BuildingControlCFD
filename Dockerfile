# OpenFOAM v2412 (official image, amd64 and arm64) + Python for running the simulation batch in a container.
# The project folder is mounted at run time, see experiments/gnot_openfoam_rans/README_batch.md.
#   docker build -t classroom-cfd .
FROM opencfd/openfoam-default:2412

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Python packages in a venv; PyTorch as CPU version (only used to write the cases, the GPU steps run outside)
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# fixed name for OpenFOAM's etc/bashrc, wherever the image has it (found by foam_env.sh via FOAM_BASHRC)
RUN B="$(ls -d /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2> /dev/null | head -1)" \
 && [ -n "$B" ] || { echo "OpenFOAM etc/bashrc not found in the image"; exit 1; } \
 && ln -s "$B" /etc/openfoam-bashrc

# one thread per process: without this every simulation starts one thread per core (load 400 on a 20-core machine)
ENV PATH=/opt/venv/bin:$PATH \
    FOAM_BASHRC=/etc/openfoam-bashrc \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    HOME=/tmp

WORKDIR /work/experiments/gnot_openfoam_rans
ENTRYPOINT []
CMD ["bash"]
