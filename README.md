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
    ```
    Hello darkness, my old friend, the sun. He is very hot and he needs to cool down. He looks around and sees a big tree. He thinks it looks like a good place to rest.
    He climbs up the tree and looks around. He sees a big, green tree with lots of leaves. He thinks it looks like a good place to rest. He climbs up the tree and sits on a branch. He feels the cool breeze on his face.
    He looks around and sees a little girl. She is playing with her doll. She has long hair and a pink dress. She looks at him and smiles. She says, "Hello, mister. Do you like my tree?"
    The old man nods and says, "Yes, I do. It is very nice. Do you want to play with me?"
    The little girl nods and says, "Yes, I do. I like your tree. It is very big and green. Can I sit with you?"
    The old man says, "Sure, you can sit with me. But be careful, don't touch my tree. It is very old and fragile. It can break easily."
    The little girl says,
    ```

## References
- [`llama2.c`](https://github.com/karpathy/llama2.c)
- [`llama2.rs`](https://github.com/gaxler/llama2.rs)
- [`llama2.c-for-dummies`](https://github.com/RahulSChand/llama2.c-for-dummies)
