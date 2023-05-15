# TextSummarizer
This is a repository for our NLP project on text summarization. We plan to implement abstractive text summarization and will be using the "CNN/Daily Mail" dataset to train our solution, which will be built on a transformer model.

We find this project particularly interesting because, in this age of information overload, filtering through large amounts of text to extract the main ideas can be very useful. This approach can speed up learning and information retrieval, making it valuable not only for personal use, but also for universities, publishing companies, and search engines.

Group Members: Carl Solli, Ashkan Aledavoud, Ali Risheh, Abhishek Rajput, Syam Jason Bonela

## Getting Started
Here is the instructions on how to work with the repository

### Running the model
To see the model in action you can run `python3 main.py example.txt` which will take the `example.txt` file provided and summarize it. 

You can replace `example.txt` with a text file of your choosing.

### Training
To train the model you can run `python3 training.py` this will run the training loop and create the model weights for you.

### Testing
--

## Different Parts Of Repo

### Preproccesing
`preprocess.py`

This is where we keep the code related to preprocessing. We're using the "CNN/Daily Mail" dataset which we tokenize, preprocess and put in buckets for a batched generator. 

### Training
`training.py`

In this file we keep our training loop. 

### Attention
`attention.py`

We've grouped the functions related to attention for our model here.

### Model
`model.py`

This is where we define our transformer model.

### Main
Our `main.py` is what runs the file summarizer program. 
