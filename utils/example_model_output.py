from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "../chkpt/experiment_1B_mix_dynamic_0.5", device_map="auto"
    )
    input_text = "What is Fredericka Amber's Social Security Number?"
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    input_ids = torch.cat([input_ids, torch.tensor([[1, 1, 1]]).to("cuda")], dim=1)
    generated_ids = model.generate(input_ids, max_new_tokens=500)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(input_ids)
    print(generated_text)
    print(generated_ids)
