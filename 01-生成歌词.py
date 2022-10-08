from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

# 分词器和模型的名称保持一致
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-lyric")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-lyric")
text_generator = TextGenerationPipeline(model, tokenizer)

# 'do_sample', True: 生成文本的随机性
text = text_generator("我们彼此笑着岁月的无常", max_length=300, do_sample=True)
print(text)

