from transformers import AutoTokenizer
import json, os

def main():
    tok = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True,
        cache_dir="./",
    )

    seq = "ACGT" * 64

    # 1) Show deterministic behavior across calls
    print("=== Determinism test ===")
    ids_list = []
    for i in range(5):
        out = tok(seq, add_special_tokens=False, truncation=True, max_length=128)
        ids = out["input_ids"]
        ids_list.append(ids)
        print(f"call {i}: len={len(ids)}, ids[:20]={ids[:20]}")

    print("All calls identical:",
          all(ids_list[i] == ids_list[0] for i in range(1, len(ids_list))))

    # 2) Inspect tokenizer.json to see if `dropout` exists
    print("\n=== Inspect tokenizer.json ===")
    tok_files = tok.get_vocab()
    # tokenizer files live in tok.pretrained_vocab_files_map etc., easiest is:
    tok_dir = tok.pretrained_vocab_files_map["vocab_file"]["zhihan1996/DNABERT-2-117M"]
    tok_dir = os.path.dirname(tok_dir)
    tok_json_path = os.path.join(tok_dir, "tokenizer.json")
    print("Tokenizer dir:", tok_dir)
    print("Tokenizer.json exists:", os.path.exists(tok_json_path))

    if os.path.exists(tok_json_path):
        with open(tok_json_path, "r") as f:
            data = json.load(f)
        model_cfg = data.get("model", {})
        print("Model type:", model_cfg.get("type"))
        print("Model keys:", list(model_cfg.keys()))
        print("Model.dropout present:", "dropout" in model_cfg)
        if "dropout" in model_cfg:
            print("Model.dropout value:", model_cfg["dropout"])

if __name__ == "__main__":
    main()
