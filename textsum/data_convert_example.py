"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import nltk.data
import struct
import sys
import csv
import chardet
import time

import tensorflow as tf
from tensorflow.core.example import example_pb2

csv.field_size_limit(sys.maxsize)

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')

def _binary_to_text():
  reader = open(FLAGS.in_file, 'rb')
  writer = open(FLAGS.out_file, 'w')
  while True:
    len_bytes = reader.read(8)
    if not len_bytes:
      sys.stderr.write('Done reading\n')
      return
    str_len = struct.unpack('q', len_bytes)[0]

    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    print(tf_example_str)
    tf_example = example_pb2.Example.FromString(tf_example_str)
    examples = []
    for key in tf_example.features.feature:
      examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
    writer.write('%s\n' % '\t'.join(examples))
  reader.close()
  writer.close()


def add_text_tags(text, min_len = 40, max_len = 10000, encode=True):
    """
    add document, paragraph, and sentence tags
    remove sentences fewer than filter_len characters
    """
    enc = chardet.detect(text)
    if enc["encoding"] != "ascii":
       text = text.decode("utf8").encode("ascii", errors="ignore") 

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    text_sents = sent_detector.tokenize(text)
    sentences = [sentence for sentence in text_sents if len(sentence) > min_len and len(sentence) < max_len]
    tagged_sentences = "<d> <p> <s> " + " </s> <s> ".join(sentences) + " </s> </p> </d>"
    #print tagged_sentences
    if encode:
      
      return  tagged_sentences.encode("utf8")
    return tagged_sentences
  
    
def _text_to_binary():
  writer = open(FLAGS.out_file, 'wb')
  start = time.time()
  with open(FLAGS.in_file, 'r') as csvfile: 
    csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    
    for rowcount, row in enumerate(csvreader):
      if rowcount % 10000 == 0:
        end = time.time()
        print "--  {} rows processed. ".format(rowcount)

      tf_example = example_pb2.Example()
      claim = add_text_tags(row[3])
      novelty = add_text_tags(row[4])
      publisher = "NULL"
      tf_example.features.feature['article'].bytes_list.value.extend([claim])
      tf_example.features.feature['abstract'].bytes_list.value.extend([novelty])
      
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))
  writer.close()


def main(unused_argv):
  assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
  if FLAGS.command == 'binary_to_text':
    _binary_to_text()
  elif FLAGS.command == 'text_to_binary':
    _text_to_binary()


if __name__ == '__main__':
  tf.app.run()


