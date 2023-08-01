# llama2.zig
Llama 2 inference in Zig

## How to run
1. Start and get inside the Docker container:
    ```bash
    cd infra-dev/
    docker-compose up -d
    docker-compose exec -it llama2 bash
    ```
2. Download models:
    ```bash
    curl -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -o ../models/TinyLlama-15M.bin
    ```
    See [`model.py`](https://github.com/karpathy/llama2.c/blob/f61807d/model.py#L317) for more details about how the `.bin` file was exported.
3. Inference:
    ```bash
    zig build run -- ../models/TinyLlama-15M.bin ../llama2.c/tokenizer.bin
    ```

## References
- [`llama2.c`](https://github.com/karpathy/llama2.c)
- [`llama2.rs`](https://github.com/gaxler/llama2.rs)
- [`llama2.c-for-dummies`](https://github.com/RahulSChand/llama2.c-for-dummies)
