FROM duckietown/challenge-aido_lf-baseline-duckietown:daffy-amd64
RUN pip3 install torch torchvision && pip3 uninstall dataclasses -y
COPY exercise_ws/src/obj_det/include/model temp_model
RUN cd temp_model && python3 -c "from model_sol import Wrapper; wrapper = Wrapper();"
