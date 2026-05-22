from rec_rlp.parse import merge_chains, parse_item_ids


def generate_vllm(prompts: list[str], cfg: dict) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model_path"],
        dtype=cfg["dtype"],
        max_model_len=cfg["max_model_len"],
    )
    params = SamplingParams(
        temperature=cfg["temperature"],
        max_tokens=cfg["max_new_tokens"],
        n=cfg["num_chains"],
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    texts = []
    for out in outputs:
        chains = [parse_item_ids(c.text, cfg["max_items"]) for c in out.outputs]
        if len(chains) == 1:
            texts.append(", ".join(chains[0]))
        else:
            texts.append(", ".join(merge_chains(chains, cfg["max_items"])))
    return texts


def generate_hf(prompts: list[str], cfg: dict) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16,
        device_map="auto",
    )
    texts = []
    for prompt in prompts:
        chains = []
        for _ in range(cfg["num_chains"]):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            chains.append(parse_item_ids(text, cfg["max_items"]))
        texts.append(", ".join(merge_chains(chains, cfg["max_items"])))
    return texts


def generate(prompts: list[str], cfg: dict) -> list[str]:
    if cfg["backend"] == "vllm":
        return generate_vllm(prompts, cfg)
    return generate_hf(prompts, cfg)
