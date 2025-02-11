import torch
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

i = str(input("Enter the text to be translated; "))

sentences = []
sentences.append(i)


#x = int(input("Enter 0 if ENG-Other; Enter 1 if Other to ENG; "))

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="tel_Telu")
batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

with tokenizer.as_target_tokenizer():
    # This scoping is absolutely necessary, as it will instruct the tokenizer to tokenize using the target vocabulary.
    # Failure to use this scoping will result in gibberish/unexpected predictions as the output will be de-tokenized with the source vocabulary instead.
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

outputs = ip.postprocess_batch(outputs, lang="tel_Telu")
print(outputs)

#>>> ['यह एक परीक्षण वाक्य है।', 'यह एक और लंबा अलग परीक्षण वाक्य है।', 'कृपया 9876543210 पर एक एस. एम. एस. भेजें और 15 अक्टूबर, 2023 तक newemail123@xyz.com पर एक ईमेल भेजें।']

['ప్రస్తుతం, నేను ఈ ప్రాజెక్ట్లో పని చేస్తున్నాను. ']

#ஹைலோ மைஂ நஂதந
#வணக்கம் நான் நந்தன்.