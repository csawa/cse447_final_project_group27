FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/predict.cpp .
COPY model.bin .
COPY predict.sh .

RUN g++ -O3 -march=native predict.cpp -o predictor

RUN chmod +x predict.sh

ENTRYPOINT ["./predict.sh"]