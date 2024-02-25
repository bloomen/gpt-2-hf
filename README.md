GPT-2 Performance Considerations
--------------------------------

# Objective

Improve the decode runtime speed on the CPU of the original GPT-2 implemention found at: https://github.com/openai/gpt-2

# Approach

After reading the GPT-2 paper and blog posts, the first thing I did was fork and clone the original gpt-2 to my local Linux workstation (Ubuntu22.04). I noticed there's a docker setup but I wanted to see if it can run using the latest Tensorflow 2.X toolchain using my host Python env. It unfortunately is not compatible as it was written with Tensorflow 1.12. So I ended up creating a docker image using the provided `Dockerfile.cpu` file while only downloading the smallest GPT-2 124M pretrained model to save on time. I also had to upgrade Tensorflow to 1.15 for it to run without errors.

After being able to generate text, I then explored the code base and added code for timing the model inference step within the `interactive_conditional_samples.py` script. My code changes can be found at https://github.com/bloomen/gpt-2 on the `inference_perf` branch. I am running the script like this:
```
python3 src/interactive_conditional_samples.py -s 42 -n 1 --top_k=40 --top_p=1 -l 100 --temperature 1
```
where the chosen parameters appear to be reasonable defaults from reading the code doc. Larger input and output lengths generally lead to higher latencies.

This is what a sample output looks like:
```
Model prompt >>> What's the capital city of the USA? The answer is:
======================================== SAMPLE 1 ========================================
 New York City. In the 1800s, there were only about six cities with a population equal to 100,000 inhabitants. Today, there are more than 50 cities each with more than 20,000 inhabitants. There are just 8 city cores in the United States of America, which are almost 30 times smaller than any city in the world.

To help solve this problem we propose to combine this unique network of 10 and 10 cities into one large urban environment across all of New York City. This
Took: 4.312711715698242
================================================================================
```
With this I got my first reference number of 4.3sec inference time. To get a better representation I ran the same script 10 times with the same prompt and received a minimum inference runtime of 3.4sec. 

# Profiling

Now having some initial numbers I wanted to find out where most runtime is spent. Using statistical profiling via `pyinstruments` I found that, as expected, almost all runtime is spent within `session.run`. It quickly dawned on me that I would need to most likely improve upon Tensorflow itself to see significant gains.

Given that this is using an ancient Tensorflow chances are there are newer implementations of GPT-2 that are signifcantly faster. This then led me to explore what's on Hugging Face.

# Using a modern model implementation from Hugging Face

I quickly came across the `GPT2LM*` classes from Hugging Face (https://huggingface.co/openai-community/gpt2) which seemed to provide an easy-to-use interface for GPT-2. My initial implementation looked like this:
```python
set_seed(42)

model_id = 'openai-community/gpt2'

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

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
```
which uses the PyTorch backend. I also experimented with TensorFlow using the `TFGPT2LMHeadModel` class but that was about a factor of 4 slower. Unclear as to why TensorFlow performs so much worse here.

Using the same hyperparameters and prompt as with the original GPT-2, I received a minimum runtime across 10 runs of 1.7sec. A sample output:
```
What's the capital city of the USA? The answer is: Manhattan. A massive building that lies about 200 feet above sea level, it is in the centre of Manhattan. It has become the third highest building in the world, with a total US$5 trillion dollar worth of homes, stores and offices. The site is not just overpriced but overpriced.

The site is the site of two major mass transit centers such as Chicago, Detroit, San Francisco, Washington DC and Singapore.
Took: 1.7931824207305908
```
As you can see, the output differs from the output of the original GPT-2 even though parameters are the same. I am guessing this could be because of slightly different implementations or additional optimizations that PyTorch makes use of. However, assuming there are no bugs the quality of both model implementations should be the same. Of course, for a production use case this would have to be verified.

# Quantization with float16

I found that 1.7sec is still relatively slow so I searched the web for approaches on how to improve LLM latency. One recurring theme is quantization (https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic) which refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision. In my case, PyTorch allowed me to use `float16` or `qint8` for dynamic quantization. `qint8` only led to garbage output so I quickly discarded it. `float16` on the other hand gave me a slight speed improvement while inference quality did not seem to be affected. However, I only tested this with a limited number of prompts. Again for production, there would have to be some large verification making sure that quantization doesn't affect quality signifcantly. It's a trade-off.

In summary, using PyTorch's `float16` I am getting a new minimum runtime of 1.5 sec. It is achieved via the following line of code:
```python
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16)
```

Reading about quantization I learned that it works because well-trained deep neural nets are quite robust to noise which is essentially what is introduced when model weights are quantized. It's a form of compression so the resulting model will be smaller in size.

# Fast tokenization

The other low-hanging fruit I came across is fast tokenization (https://huggingface.co/learn/nlp-course/en/chapter6/3). This is using natively compiled extensions for encoding and decoding tokens. I tried it out using the `GPT2TokenizerFast` class. It did not affect inference runtime in my testing but also didn't affect quality. It seems an established method and so I kept it in my final code.

# Conclusions

The final code I came up with is in `run.py`. It doesn't take parameters and will simply evaluate the smallest GPT-2 model with hardcoded prompt and parameters. It uses float16 quantization and fast tokenization. This reduces the GPT-2 inference runtime from 3.4 to 1.5 seconds. I think this would be a magnitude too slow for a production use case.

# Outlook

Due to time constraints, I only ran tests against the smallest GPT-2 (124M) model. Ideally, my tests should be repeated for the larger models as well. However, I think this greatly depends on the use case. E.g., in the cloud the largest model is probably great but for an edge device a smaller model may be more appropriate due to resource constraints.

My testing was also limited to a restricted number of prompts but ideally this should be repeated using some sort of automated framework for model validation.

During my research I found a few other avenues to explore to improving inference latency (e.g. https://www.tensorflow.org/lite/performance/model_optimization, https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices). This includes:
* Pruning, i.e. remove unnecessary connections in the model
* Simplify the model architecture, i.e. Occams Razor
* Optimized deep learning libs
* Compressing the model into different representations
* Optimized tokenization
* Exploit parallelism

Any of these could affect model quality so ideally, before optimizing, one would need an automated way to ensure model quality isn't significantly affected.

Finally, in a production scenario one should also be considering multiple users. In this context and in addition to latency, other performance metrics such as Time To First Token (TTFT), Time Per Output Token (TPOT), and throughput would be worth considering.
