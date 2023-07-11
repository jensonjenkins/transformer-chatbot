import numpy as np
import tensorflow as tf

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

def translate_token(sentence, start_id, end_id, max_length=100):
  enc_inputs = tokenizer(sentence)[tf.newaxis]
  dec_inputs = np.zeros((1, max_length))
  dec_inputs[0][0] = start_id
  for i in range(max_length):
    dec_outputs = tf.argmax(transformer((enc_inputs, dec_inputs), training=False), axis=-1)
    # print(dec_outputs)
    dec_inputs[0][i+1] = dec_outputs[0][i]
    if dec_inputs[0, i]==end_id:
      break
  return dec_inputs

def detokenize(tokens, end_id):
  output = ""
  for i in range(1, len(tokens[0])):
    if tokens[0][i] == end_id:
      break
    else:
      output+=tokenizer.get_vocabulary()[tokens[0][i]]+" "
  return output

def run():
  userI =""
  while True:
    userI = input("Question: ")
    if userI == "breakfromloop0":
      print("Session Ended")
      break
    print("Generated Answer: ", end="")
    print(detokenize(translate_token(userI)))
    print()

# FOR TOKENIZER
def add_start(ragged):
  reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
  START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  return tf.concat([starts, ragged], axis=1)


def add_end(ragged):
  reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
  END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
  count = ragged.bounding_shape()[0]
  ends = tf.fill([count,1], END)
  return tf.concat([ragged, ends], axis=1)
