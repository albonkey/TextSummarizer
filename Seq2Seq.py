from Seq2SeqModel import build_seq2seq_model_with_just_lstm
from preprocessing.Seq2SeqPreprocess import create_inputs

x_vocab_size, x_embedding_matrix, y_vocab_size, y_embedding_matrix = create_inputs()

seq2seq = build_seq2seq_model_with_just_lstm(
    x_embedding_matrix, y_embedding_matrix,
    x_vocab_size=x_vocab_size, y_vocab_size=y_vocab_size
)