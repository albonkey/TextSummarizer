import trax
from rouge import Rouge
from model import TransformerLM
from decoding.decoders import decode, beam_search_decode


def evaluate_rouge(reference_sentences, generated_summary):
    """Evaluate ROUGE scores between a reference and a generated summary.

    Args:
        reference_sentences (list): list of reference sentences.
        generated_summary (str): generated summary.

    Returns:
        dict: ROUGE scores.
    """
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_sentences)
    return scores[0]  # Return the first item since there's only one summary

def get_article_and_summary(stream):
    for (article, summary) in stream:
        yield article, summary


eval_stream_fn = trax.data.TFDS(
    'cnn_dailymail',
    keys=('article', 'highlights'),
    train=False
)

eval_data = get_article_and_summary(eval_stream_fn())

# Init Model
model = TransformerLM(mode='eval')
model.init_from_file('model.pkl.gz', weights_only=True)

# Evaluating our model with Rouge
evaluations = []

for i in range(3):
    article, summary = next(eval_data)
    generated_summary = decode(article, model, method='sampling')
    rouge_scores = evaluate_rouge(str(summary), generated_summary)
    evaluations.append(
        (summary, generated_summary, rouge_scores)
    )

# Printing the result of the Rouge evaluation
total_f = 0
total_precision = 0
total_recall = 0
for i, eval in enumerate(evaluations):
    print('Article ' + str(i + 1))
    print(eval[2]['rouge-1'])

    total_f += eval[2]['rouge-1']['f']
    total_precision += eval[2]['rouge-1']['p']
    total_recall += eval[2]['rouge-1']['r']

print('Average F-Score: ' + str(total_f/len(eval)))
print('Average Precision: ' + str(total_precision/len(eval)))
print('Average Recall: ' + str(total_recall/len(eval)))