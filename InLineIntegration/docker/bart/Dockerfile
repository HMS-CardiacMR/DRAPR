# ----- First stage to build BART -----
FROM python:3.9-slim AS bart_build
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN  apt-get update && apt-get install -y git cmake g++ libfftw3-dev liblapacke-dev libpng-dev gfortran
RUN  mkdir -p /opt/code

# BART (static linked)
RUN cd /opt/code                                   && \
    git clone https://github.com/mrirecon/bart.git --branch v0.7.00 && \
    cd bart                                        && \
    make SLINK=1 -j $(nproc)                       && \
    make install

# ----- Main stage without build dependencies -----
# Re-use already built Docker image, but the contents of \docker\Dockerfile can also be
# recapitulated here instead to ensure the latest build
FROM kspacekelvin/fire-python
ENV PYTHONPATH=/opt/code/bart/python

# Copy BART from previous stage
COPY --from=bart_build /usr/local/bin/bart             /usr/local/bin/
COPY --from=bart_build /usr/local/lib/bart/commands    /usr/local/lib/bart/commands
COPY --from=bart_build /usr/local/share/doc/bart       /usr/local/share/doc/bart
COPY --from=bart_build /opt/code/bart                  /opt/code/bart
