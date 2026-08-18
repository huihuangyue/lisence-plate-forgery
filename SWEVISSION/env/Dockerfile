FROM vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
RUN /root/miniconda3/envs/sandbox-runtime/bin/pip install ipython==8.18.1 --force-reinstall

# Ensure the conda env's Python is on PATH by default
ENV PATH="/root/miniconda3/envs/sandbox-runtime/bin:${PATH}"

WORKDIR /mnt/data