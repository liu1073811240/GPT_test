from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline


tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-ancient")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-ancient")
text_generator = TextGenerationPipeline(model, tokenizer)
text = text_generator("当是时", max_length=100, do_sample=True)
print(text)
