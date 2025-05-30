import tiktoken

encoder= tiktoken.encoding_for_model('gpt-4o')

print("Vocabulary size:", encoder.n_vocab) # 200019 tokens in gpt-4o

text= "The quick brown fox jumps over the lazy dog."
tokens = encoder.encode(text)

print("Tokens:", tokens) # Tokens: [976, 4853, 19705, 68347, 65613, 1072, 290, 29082, 6446, 13]

my_tokens= [976, 4853, 19705, 68347, 65613, 1072, 290, 29082, 6446, 13]
decoded_text=encoder.decode(my_tokens)
print("Decoded text:", decoded_text) # Decoded text: The quick brown fox jumps over the lazy dog.
