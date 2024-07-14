import torch
import transformers
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig
from huggingface_hub import notebook_login, whoami

def Llama2(prompt: str):
  if whoami() == False:
    notebook_login()

  modelName = "meta-llama/Llama-2-7b-chat-hf"
  tokenizerName = "hf-internal-testing/llama-tokenizer"
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4", # Normal Float 4
      bnb_4bit_compute_dtype=torch.float16,
  )

  model = AutoModelForCausalLM.from_pretrained(
          modelName,
          quantization_config=bnb_config,
          device_map="auto"
  )
  tokenizer = LlamaTokenizerFast.from_pretrained(
      tokenizerName,
      torch_dtype=torch.float16,
      device_map="auto"
  )
  pipeline = transformers.pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
  )

  sequences = pipeline(
      prompt,
      top_k=40,
      top_p=0.95,
      eos_token_id=tokenizer.eos_token_id,
      batch_size=1,
  )

  for seq in sequences:
      print(seq['generated_text'])

