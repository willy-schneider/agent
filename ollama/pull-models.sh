#!/bin/bash

MODELS=${OLLAMA_MODELS:-"qwen2.5:3b yxchia/multilingual-e5-base:latest"}

ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama server..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 1
done
echo "Ollama ready"

for model in $MODELS; do
  if ollama list | grep -q "$model"; then
    echo "Model already exists: $model"
  else
    echo "Pulling model: $model"
    ollama pull "$model"
  fi
done

wait $SERVER_PID
