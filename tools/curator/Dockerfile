# Curator image: the GSR ROCm base plus diffusers (FLUX) and the web server.
FROM jng-gsr:rocm7
RUN pip install --no-cache-dir \
    diffusers transformers accelerate sentencepiece protobuf peft \
    fastapi uvicorn
