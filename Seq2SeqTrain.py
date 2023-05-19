from Seq2SeqModel import build_seq2seq_model_with_just_lstm
from Seq2SeqModel import build_seq2seq_model_with_bidirectional_lstm
from Seq2SeqModel import build_hybrid_seq2seq_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocessing.Seq2SeqPreprocess import create_inputs

x_vocab_size, x_embedding_matrix, y_vocab_size, y_embedding_matrix, x_train_padded, y_train_padded, x_val_padded, y_val_padded = create_inputs()

seq2seq = build_seq2seq_model_with_just_lstm(
    x_embedding_matrix, y_embedding_matrix,
    x_vocab_size=x_vocab_size, y_vocab_size=y_vocab_size
)

seq2seqbidirectional = build_seq2seq_model_with_bidirectional_lstm(
    x_embedding_matrix, y_embedding_matrix,
    x_vocab_size=x_vocab_size, y_vocab_size=y_vocab_size
)

seq2seqhybrid = build_hybrid_seq2seq_model(
    x_embedding_matrix, y_embedding_matrix,
    x_vocab_size=x_vocab_size, y_vocab_size=y_vocab_size
)
model = seq2seq['model']

callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.000001, verbose=1),
]

history = model.fit(
    [x_train_padded, y_train_padded[:, :-1]],
    y_train_padded.reshape(y_train_padded.shape[0], y_train_padded.shape[1], 1)[:, 1:],
    epochs = 2,
    # batch_size=128 * tpu_strategy.num_replicas_in_sync,
    batch_size = 128,
    callbacks=callbacks,
    validation_data=(
        [x_val_padded, y_val_padded[:, :-1]],
        y_val_padded.reshape(y_val_padded.shape[0], y_val_padded.shape[1], 1)[:, 1:]
    )
)