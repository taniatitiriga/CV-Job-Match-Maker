@echo off
start "" /B .\llama-server -m ..\models\mxbai-embed-large-v1-f16.gguf --host 0.0.0.0 -ngl -1 --embedding --port 8081
start "" /B .\llama-server -m ..\models\phi-4-mini-instruct-Q5_K_M.gguf --host 0.0.0.0 -ngl -1 --port 8080
