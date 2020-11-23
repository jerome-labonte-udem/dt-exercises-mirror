FROM duckietown/challenge-aido_lf-baseline-duckietown:daffy-amd64
RUN pip3 install torch torchvision && pip3 uninstall dataclasses -y
