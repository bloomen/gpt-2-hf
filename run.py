import torch
import time
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, set_seed


def main():
    set_seed(42)

    model_id = 'openai-community/gpt2'

    # Load tokenizer and model
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    assert tokenizer.is_fast
    model = GPT2LMHeadModel.from_pretrained(model_id)

    # Perform float16 quantization
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16)

    # Model evaluation
    text = "What's the capital city of the USA? The answer is:"
    encoded_input = tokenizer.encode(text, return_tensors='pt')
    start = time.time()
    output = model.generate(encoded_input, 
                            max_length=100, 
                            num_return_sequences=1, 
                            do_sample=True, 
                            temperature=1, 
                            top_k=40,
                            top_p=1,
                            pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    print("Took:", time.time() - start)


if __name__ == '__main__':
    main()
