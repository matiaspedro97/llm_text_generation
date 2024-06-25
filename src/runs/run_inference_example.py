import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# prompt template
def generate_inference_prompt(context, question):
    return f"""### Instruction: Please answer to the question based on the context information provided. If you don't know the answer, please just say you don't know it, don't try to make an answer from that.\n
            ### Context:
            {context.strip()}\n

            ### Question:
            {question.strip()}

            ### Answer:
            
            """.strip()

# context to answer
context = """
No estrangeiro, em que dias decorre a votação?
A votação tem lugar no dia anterior ao marcado para a eleição e no próprio dia da eleição. 
"""

# question to ask
question = """
I live in the Netherlands. Am I able to vote in these elections in another european country, e.g., Prague, in the portuguese embassy? 
"""

# loading model
model = AutoModelForCausalLM.from_pretrained(
'pedromatias97/little-llama2-ft-qa'
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
'pedromatias97/little-llama2-ft-qa'
)

# pipeline
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# generate prompt
prompt = generate_inference_prompt(context, question)

# generate text
sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=10, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)

# print result
print(sequences[0]['generated_text'])

### output: 40 per cent that of Great Britain
