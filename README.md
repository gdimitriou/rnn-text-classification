# Description:
This is a Recurrent Neural Network that performs text toxicity classification

    Python (3.6)
    Tensorflow (2.1.0)
    Keras (2.3.1)
    Pillow (7.1.1)
    matplotlib (3.2.1)
    np (1.0.2)
    scikit-learn (0.22.2.post)
    numpy (1.18.2)
    os

Instructions to run the program:

    1. Clone the project from: https://github.com/gdimitriou/rnn-text-classification.git.
    2. Import it to your favorite IDE.
    3. Download the dependencies.
    4. Download the embeddings from: 
        1. https://nlp.stanford.edu/projects/glove/
        2. https://fasttext.cc/docs/en/english-vectors.html
    5. Import the embeddings manually under resources directory.    
    5. Run the text-classifier.py.

Expecting output:
    The model's accuracy scores over 98%.
    The model predicts a probability for each variable imported in the dataset.
    The output should look like sample_submission.txt
