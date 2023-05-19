import numpy as np
from helpers import tokenize, detokenize, measure_time

def next_symbol_greedy(cur_output_tokens, model):
    """Returns the next symbol for a given sentence.

    Args:
        cur_output_tokens (list): tokenized sentence with EOS and PAD tokens at the end.
        model (trax.layers.combinators.Serial): The transformer model.

    Returns:
        int: tokenized symbol.
    """
    
    # current output tokens length
    token_length = len(cur_output_tokens)
    # calculate the minimum power of 2 big enough to store token_length
    # HINT: use np.ceil() and np.log2()
    # add 1 to token_length so np.log2() doesn't receive 0 when token_length is 0
    padded_length = 2**int(np.ceil(np.log2(token_length + 1)))

    # Fill cur_output_tokens with 0's until it reaches padded_length
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :] # Don't replace this None! This is a way of setting the batch dim

    # model expects a tuple containing two padded tensors (with batch)
    output, _ = model((padded_with_batch, padded_with_batch)) 
    # HINT: output has shape (1, padded_length, vocab_size)
    # To get log_probs you need to index output wih 0 in the first dim
    # token_length in the second dim and all of the entries for the last dim.
    log_probs = output[0, token_length, :]
    
    return int(np.argmax(log_probs))

def next_symbol_sampling(cur_output_tokens, model):
    """Returns the next symbol for a given sentence.

    Args:
        cur_output_tokens (list): tokenized sentence with EOS and PAD tokens at the end.
        model (trax.layers.combinators.Serial): The transformer model.

    Returns:
        int: tokenized symbol.
    """
    token_length = len(cur_output_tokens)
    padded_length = 2 ** int(np.ceil(np.log2(token_length + 1)))

    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :]

    output, _ = model((padded_with_batch, padded_with_batch))
    log_probs = output[0, token_length, :]
    
    # Convert log probabilities to probabilities using softmax
    probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
    
    # Sample the next symbol based on the probabilities
    next_symbol = int(np.random.choice(len(probs), p=probs))
    
    return next_symbol

@measure_time
def decode(input_sentence, model, method='greedy', tokenize=tokenize, detokenize=detokenize, vocab_dir='vocab_dir/', verbose=False):
    """Decode function.

    Args:
        input_sentence (string): a sentence or article.
        model (trax.layers.combinators.Serial): Transformer model.

    Returns:
        string: summary of the input.
    """
    
    # Use tokenize()
    cur_output_tokens = tokenize(input_sentence) + [0]    
    generated_output = [] 
    cur_output = 0 
    EOS = 1 
    
    while cur_output != EOS:
        # Get next symbol
        cur_output = None 
        if (method == 'greedy'):
            cur_output = next_symbol_greedy(cur_output_tokens, model)

        elif(method == 'sampling'):
            cur_output = next_symbol_sampling(cur_output_tokens, model)

        # Append next symbol to original sentence
        cur_output_tokens.append(cur_output)
        # Append next symbol to generated sentence
        generated_output.append(cur_output)
        
        if verbose:
            print(detokenize(generated_output, vocab_dir=vocab_dir))
    
        
    return detokenize(generated_output, vocab_dir=vocab_dir)

def beam_search_decoder(cur_output_tokens, model, beam_width):
    """Beam search decoding function for the next symbol.

    Args:
        cur_output_tokens (list): Tokenized sentence with EOS and PAD tokens at the end.
        model (trax.layers.combinators.Serial): Transformer model.
        beam_width (int): Number of candidate symbols to consider.

    Returns:
        list: Candidate symbols.
    """
    token_length = len(cur_output_tokens)
    padded_length = 2 ** int(np.ceil(np.log2(token_length + 1)))
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :]

    output, _ = model((padded_with_batch, padded_with_batch))
    log_probs = output[0, token_length, :]

    # Convert log probabilities to probabilities using softmax
    probs = np.exp(log_probs) / np.sum(np.exp(log_probs))

    # Select the top-k candidate symbols based on probabilities
    topk_symbols = np.argsort(probs)[-beam_width:][::-1]

    return topk_symbols

@measure_time
def beam_search_decode(input_sentence, model, beam_width, tokenize=tokenize, detokenize=detokenize, vocab_dir='vocab_dir/', verbose=False):
    """Beam search decoding function.

    Args:
        input_sentence (string): A sentence or article.
        model (trax.layers.combinators.Serial): Transformer model.
        beam_width (int): Number of candidate sequences to consider.
        tokenize (function): Tokenization function.
        detokenize (function): Detokenization function.
        vocab_dir (string): Directory path for vocabulary.
        verbose (bool): Flag to print intermediate decoding steps.

    Returns:
        string: Summary of the input.
    """
    # Use tokenize()
    cur_output_tokens = tokenize(input_sentence) + [0]
    generated_outputs = [[]] * beam_width
    EOS = 1

    while len(generated_outputs[0]) == 0 or generated_outputs[0][-1] != EOS:
        beam_candidates = []

        for output_idx in range(beam_width):
            # Get next symbols for each candidate sequence
            cur_output = generated_outputs[output_idx]
            cur_output_tokens_temp = cur_output_tokens + cur_output

            # Use beam search decoder for next symbol selection
            next_symbols = beam_search_decoder(cur_output_tokens_temp, model, beam_width)

            for symbol_idx, next_symbol in enumerate(next_symbols):
                # Add the next symbol to candidate sequence
                candidate_sequence = cur_output + [next_symbol]
                beam_candidates.append((output_idx, candidate_sequence))

        # Update the generated outputs
        generated_outputs = [generated_outputs[output_idx] + [candidate_sequence[-1]] for output_idx, candidate_sequence in beam_candidates]

        if verbose:
            for idx, output in enumerate(generated_outputs):
                print(detokenize(output, vocab_dir=vocab_dir))

    return detokenize(generated_outputs[0], vocab_dir=vocab_dir)