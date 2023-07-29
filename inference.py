import tensorflow as tf
import tensorflow_text as text
import pandas as pd
import numpy as np
import random
from transformer import Transformer, CustomSchedule
from utils import masked_loss, masked_accuracy

VOCAB_SIZE = 15800
MAX_LENGTH = 100
EMBED_SIZE = 128

tokenizer = tf.saved_model.load("sub_word_tokenizer")

q = tf.zeros((64, 100))

transformer = Transformer(num_layers=6, embed_size=EMBED_SIZE,num_heads=8, dff=128, vocab_size=VOCAB_SIZE, dropout_rate=0.1)
output = transformer((q, q))
weights = "00_LaMini_5e.h5"
transformer.load_weights(weights)

# print(transformer.summary())

lines = []

with open("vocab.txt", 'r') as file:
    for line in file:
        lines.append(line.strip())

def detokenize(tokens, end_id):
    output=""
    for i in range(1, len(tokens[0])):
        if tokens[0][i] == end_id:
            break
        elif lines[tokens[0][i]][0] == "#":
            output+=lines[tokens[0][i]][2:]+" "
        else:
            output+=lines[tokens[0][i]]+" "
    return output

def translate_token(sentence, start_id, end_id, model, max_length=100):
    enc_inputs = tokenizer.sw.tokenize([sentence])
    enc_values = enc_inputs.values
    enc_splits = enc_inputs.row_splits
    enc_inputs = tf.RaggedTensor.from_row_splits(enc_values, enc_splits).to_tensor(default_value=0, shape=(1,100))

    dec_inputs = np.zeros((1, max_length))
    dec_inputs[0][0] = start_id
    for i in range(max_length-1):
        dec_outputs = tf.argmax(model((enc_inputs, dec_inputs), training=False), axis=-1)
        dec_inputs[0][i+1] = dec_outputs[0][i]
        if dec_outputs[0, i]==end_id:
            break
        print(lines[dec_outputs[0][i]], end=" ", flush=True)
    return dec_inputs
print("\n\n=======================================================")
print("weights: " + weights[0:-3])
print("=======================================================\n\n")
userI = input(">> Q: ")
while userI != "endofquery":
    print(">> A: ", end="")
    tokenized = translate_token(userI, 2, 3, transformer)
    tokenized = tf.cast(tokenized, dtype=tf.int32)
    # print(">> A: "+ detokenize(tokenized, 3) + "\n\n")
    userI = input("\n\n>> Q: ")






