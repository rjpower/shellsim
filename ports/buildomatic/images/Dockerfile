FROM ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab

LABEL org.opencontainers.image.description="Buildomatic native SDK bootstrap inputs and Iris worker tools"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ binutils libc6-dev ninja-build pkgconf patch git tar gzip xz-utils \
      ca-certificates coreutils findutils sed grep gawk autoconf automake libtool \
    && rm -rf /var/lib/apt/lists/*

COPY --from=inputs /cmake-3.31.8-linux-x86_64.tar.gz /opt/buildomatic-inputs/cmake.tar.gz
RUN echo '630615d8e98ac33eba7fbe472626dff5c899c85af3c024585ae109166a6909d0  /opt/buildomatic-inputs/cmake.tar.gz' | sha256sum -c - \
    && mkdir -p /opt/buildomatic/native/cmake \
    && tar -xzf /opt/buildomatic-inputs/cmake.tar.gz --strip-components=1 -C /opt/buildomatic/native/cmake \
    && rm /opt/buildomatic-inputs/cmake.tar.gz \
    && mkdir -p /opt/buildomatic/native/gcc/bin /opt/buildomatic/native/gcc/lib/gcc \
    && cp -aL /usr/lib/gcc/. /opt/buildomatic/native/gcc/lib/gcc/ \
    && cp -aL /usr/libexec/gcc/. /opt/buildomatic/native/gcc/lib/gcc/ \
    && cp -L /usr/bin/gcc /opt/buildomatic/native/gcc/lib/gcc/x86_64-linux-gnu/14/gcc \
    && cp -L /usr/bin/g++ /opt/buildomatic/native/gcc/lib/gcc/x86_64-linux-gnu/14/g++ \
    && mkdir -p /opt/buildomatic/native/gcc/lib/gcc/x86_64-linux-gnu/14/include/c++/x86_64-linux-gnu \
    && cp -aL /usr/include/c++/14/. /opt/buildomatic/native/gcc/lib/gcc/x86_64-linux-gnu/14/include/c++/ \
    && cp -aL /usr/include/x86_64-linux-gnu/c++/14/. /opt/buildomatic/native/gcc/lib/gcc/x86_64-linux-gnu/14/include/c++/x86_64-linux-gnu/ \
    && printf '%s\n' '#!/usr/bin/sh' 'set -eu' \
      'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)' \
      'support="$root/lib/gcc/x86_64-linux-gnu/14"' \
      'export CPLUS_INCLUDE_PATH="$support/include/c++:$support/include/c++/x86_64-linux-gnu:$support/include/c++/backward"' \
      'exec "$support/gcc" -B"$support/" "$@"' > /opt/buildomatic/native/gcc/bin/gcc \
    && sed 's|support/gcc|support/g++|' /opt/buildomatic/native/gcc/bin/gcc > /opt/buildomatic/native/gcc/bin/g++ \
    && chmod 0555 /opt/buildomatic/native/gcc/bin/gcc /opt/buildomatic/native/gcc/bin/g++ \
    && mkdir -p /opt/buildomatic/native/pkgconf/bin /opt/buildomatic/native/pkgconf/lib /opt/buildomatic/native/ninja/bin \
    && cp -L /usr/bin/pkgconf /opt/buildomatic/native/pkgconf/lib/pkgconf \
    && cp -L /usr/lib/x86_64-linux-gnu/libpkgconf.so.3 /opt/buildomatic/native/pkgconf/lib/libpkgconf.so.3 \
    && cp -L /usr/bin/ninja /opt/buildomatic/native/ninja/bin/ninja \
    && printf '%s\n' '#!/usr/bin/sh' 'set -eu' \
      'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)' \
      'export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
      'exec "$root/lib/pkgconf" "$@"' > /opt/buildomatic/native/pkgconf/bin/pkg-config \
    && chmod 0555 /opt/buildomatic/native/pkgconf/bin/pkg-config \
    && chmod -R a-w /opt/buildomatic/native

# Caller utilities keep their exact bytes and aliases. Install them after apt,
# so package installation cannot change the bindings checked by SDK admission.
COPY --from=utilities /make /usr/bin/make
COPY --from=utilities /dash /usr/bin/dash
COPY --from=utilities /gnurm /usr/bin/gnurm
RUN chmod 0555 /usr/bin/make /usr/bin/dash /usr/bin/gnurm \
    && rm /usr/bin/sh /usr/bin/rm \
    && ln -s dash /usr/bin/sh && ln -s gnurm /usr/bin/rm \
    && echo '27c9f6d806aee15882b01c2c61848f7aa75caa14bc7b6f608ba422f9e46a7d49  /usr/bin/make' | sha256sum -c - \
    && echo 'c626229526bb58ec2d0f585f3c3ae1412e6f973b4353385042d11c38d8426917  /usr/bin/sh' | sha256sum -c - \
    && echo '0362781f855d9de6396b71af947662758970ed09946c4a0a78ff740b20f5e6a6  /usr/bin/rm' | sha256sum -c -

COPY --from=sccache /sccache /usr/local/bin/sccache
RUN chmod 0555 /usr/local/bin/sccache \
    && echo '973cb15f6a986d84ca334bbed3bbe2eb8f1ee8fd81bf9e115b8539a293bf8d59  /usr/local/bin/sccache' | sha256sum -c -

COPY --from=helper / /opt/buildomatic/python-helper/
RUN chmod -R a-w /opt/buildomatic/python-helper

# Copy selected immutable inputs only. Workspaces and prior SDK registries never
# enter this image; source archives remain available for the offline producer.
COPY --from=prepared /host/ /home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1/host/
COPY --from=prepared /store/sources/ /home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1/store/sources/
COPY --from=prepared /host-seed.prepared.json /home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1/host-seed.prepared.json
COPY --from=prepared /prepared.json /home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1/prepared.json
COPY --from=prepared /staging-proof.json /home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1/staging-proof.json
COPY --from=code / /opt/buildomatic/ports/
COPY preflight.py /opt/buildomatic/preflight.py
COPY inputs.json /opt/buildomatic/inputs.json
COPY --from=inputs /packaging-25.0-py3-none-any.whl /opt/buildomatic-inputs/packaging.whl
RUN echo '29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484  /opt/buildomatic-inputs/packaging.whl' | sha256sum -c - \
    && python3 -c 'import sysconfig,zipfile; zipfile.ZipFile("/opt/buildomatic-inputs/packaging.whl").extractall(sysconfig.get_paths()["purelib"])' \
    && rm /opt/buildomatic-inputs/packaging.whl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/buildomatic/native/cmake/bin:/opt/buildomatic/native/ninja/bin:/opt/buildomatic/native/pkgconf/bin:/usr/local/bin:/usr/bin:/bin"
RUN python3 -B /opt/buildomatic/preflight.py

CMD ["python3"]
