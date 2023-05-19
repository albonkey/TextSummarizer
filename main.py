# Import libraries
import sys

from model import TransformerLM
from helpers import decode, beam_search_decode

if len(sys.argv) < 2:
    print("Please specify the file name of the .txt file you would like to summarize.")
    sys.exit(1)

filename = sys.argv[1]

with open(filename, 'r') as file:
    file_contents = file.read()

#print("Starting to summarize %..." % filename)
model = TransformerLM(mode='eval')

# Load the pre-trained weights
model.init_from_file('./model/model.pkl.gz', weights_only=True)

print("The summarization of the file you provided is:")
print(decode(file_contents, model, verbose=True))