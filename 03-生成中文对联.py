from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline


tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-couplet")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-couplet")
text_generator = TextGenerationPipeline(model, tokenizer)
text = text_generator("[CLS]丹 枫 江 冷 人 初 去 -", max_length=25, do_sample=True)

print(text)


