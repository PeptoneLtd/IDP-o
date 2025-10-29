FROM nvcr.io/nvidia/jax:24.10-py3

RUN pip install foldcomp~=0.1.0 tables~=3.10.0 cupy-cuda12x~=13.6.0 hirola~=0.3.0 pynmrstar~=3.3.6 joblib~=1.5.2  git+https://github.com/PeptoneLtd/nerfax.git@2dd1ea0

COPY . /IDP-o/
ENV PYTHONPATH=/IDP-o/

ENTRYPOINT ["python", "/IDP-o/scripts/build_ensemble.py"]
