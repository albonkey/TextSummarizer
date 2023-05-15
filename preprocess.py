import trax
import numpy as np

# Special tokens
SEP = 0 # Padding or separator token
EOS = 1 # End of sentence token

def preprocess(stream):
    for (article, summary) in stream:
        joint = np.array(list(article) + [EOS, SEP] + list(summary) + [EOS])
        mask = [0] * (len(list(article)) + 2) + [1] * (len(list(summary)) + 1) # Accounting for EOS and SEP
        yield joint, joint, np.array(mask)

def process_dataset(dataset):
    input_pipeline = trax.data.Serial(
        # Tokenizes
        trax.data.Tokenize(vocab_dir='vocab_dir/',
                        vocab_file='summarize32k.subword.subwords'),
        # Uses function defined above
        preprocess,
        # Filters out examples longer than 2048
        trax.data.FilterByLength(2048)
    )

    return input_pipeline(dataset)

# Function that processes the data streams
def get_data_streams():
    # Importing CNN/DailyMail articles dataset
    train_stream_fn = trax.data.TFDS('cnn_dailymail',
                                    keys=('article', 'highlights'),
                                    train=True)

    # This should be much faster as the data is downloaded already.
    eval_stream_fn = trax.data.TFDS('cnn_dailymail',
                                    keys=('article', 'highlights'),
                                    train=False)


    # Apply preprocessing to data streams.
    train_stream = process_dataset(train_stream_fn())
    eval_stream = process_dataset(eval_stream_fn())

    train_input, train_target, train_mask = next(train_stream)

    assert sum((train_input - train_target)**2) == 0  # They are the same in Language Model (LM).

    # Bucketing to create batches of data
    boundaries =  [128, 256,  512, 1024]
    batch_sizes = [16,    8,    4,    2, 1]

    # Create the streams.
    train_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes)(train_stream)

    eval_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes)(eval_stream)
    
    return (train_batch_stream, eval_batch_stream)