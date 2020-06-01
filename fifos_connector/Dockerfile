FROM duckietown/aido3-base-python3:daffy


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip list

COPY . .
ENV DISABLE_CONTRACTS=1



ENTRYPOINT ["python3", "middleware.py"]
