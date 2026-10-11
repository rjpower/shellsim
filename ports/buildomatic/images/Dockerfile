FROM ghcr.io/astral-sh/uv:0.12.3 AS uv
FROM ubuntu:24.04

LABEL org.opencontainers.image.description="Buildomatic Iris worker with ordinary Ubuntu build tools"

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config patch git curl \
      ca-certificates tar gzip pigz xz-utils unzip autoconf automake libtool \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v0.18.0/sccache-v0.18.0-x86_64-unknown-linux-musl.tar.gz \
      | tar -xz --strip-components=1 -C /usr/local/bin --wildcards '*/sccache' \
    && chmod +x /usr/local/bin/sccache
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.97.1

# Runtime/code transport installs Iris into its own environment. Generators
# and the native CPython build helper are reusable image dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.13 \
    && uv venv --python 3.13 /opt/buildomatic/venv \
    && uv pip install --python /opt/buildomatic/venv/bin/python \
      packaging meson meson-python==0.16.0 cython numpy==2.3.5 \
      pybind11 setuptools setuptools-scm==8.3.1 wheel
ENV PATH="/opt/buildomatic/venv/bin:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ENV CARGO_TARGET_DIR=/app/.buildomatic/cargo-build
ENV RUSTUP_HOME=/root/.rustup
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
CMD ["python"]
