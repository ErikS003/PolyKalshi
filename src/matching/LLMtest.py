import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import torch
import time

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")

df = pd.read_csv(r"Data/candidate_series_matches.csv")
print(df.columns.tolist())
print(df[["kalshi_candidate_title_clean","polymarket_candidate_title_clean"]])
PRE_PROMPT = """You are a contract equivalence classifier.

Task:
Determine whether two prediction market contracts resolve based on the same underlying event.

Important rules:
- Ignore issuance date.
- Ignore tiny time differences such as minutes or timezone boundary phrasing.
- Focus only on whether the real-world YES-resolution event is the same.
- Only compare the resolution conditions.
- Do not use today's date.
- Do not reject equivalence just because one text says "before 2027" and another says "during 2026" if they describe the same event window.
- If the underlying action is different (visit vs hold office, meet vs pardon, etc.), label Not Equivalent.

Return exactly this format and nothing else:

Label: Not Equivalent
Reason: <12 words max>

or

Label: Equivalent
Reason: <12 words max>
"""
TOKENS = False
HF_TOKEN = "HF"

if not TOKENS:
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("hf_device_map:", getattr(model, "hf_device_map", None))


def build_prompt(contract_a: str, contract_b: str, title_a, title_b) -> list[dict]:
    user_prompt = f"""
    
    Contract A title: {title_a}, 
    Contract A description: {contract_a}
    \n0
    \n
    Contract B title: {title_b}, 
    Contract B description: {contract_b}


    Are these contracts equivalent?"""
    return [
        {"role": "system", "content": PRE_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


for i in range(0, 30):
    title_p = df.loc[i, "polymarket_candidate_title_clean"]
    title_k = df.loc[i, "kalshi_candidate_title_clean"]
    row_k = df.loc[i, "kalshi_rules_text"]
    row_p = df.loc[i, "polymarket_rules_text"]

    if pd.isna(row_k) or pd.isna(row_p):
        print(f"Row {i}: skipped because of missing rules text")
        continue

    start = time.perf_counter()

    if TOKENS:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )

        completion = client.chat.completions.create(
            model="Qwen/Qwen3.5-35B-A3B:novita",
            messages=build_prompt(str(row_k), str(row_p)),
        )

        response = completion.choices[0].message.content
        print(f"Row {i}")
        print(response)

    else:
        messages = build_prompt(str(row_k), str(row_p), str(title_k), str(title_p))

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(text, return_tensors="pt")

        if torch.cuda.is_available():
            model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        new_tokens = generated_ids[:, model_inputs["input_ids"].shape[1]:]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        print(f"Row {i}")
        print(response)
        print(f"\nActual titles: kalshi: {title_k}, \n polymarket: {title_p}")

    end = time.perf_counter()
    print(f"\nPrompt {i} took {end - start:.2f} seconds\n")