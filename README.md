# Chinese Word Segmentation Walkthrough

### Getting the data and setting up

Download ```icwb2-data.zip``` from the webpage 
http://sighan.cs.uchicago.edu/bakeoff2005/ and extract it
into the directory ```data/bakeoff/```. Also, create a directory
called ```cached``` in the repository's root directory.

### Creating vector embeddings for segmentation decisions

First, create a cached CSV "cached/vecs.train.csv" of vector embeddings for
a small word-segmented Chinese corpus (be advised that for the final step, 
there are approximately 1000 sentences to process):

    from embed import *
    from readdata import read_from_training_data
    dataset = read_from_training_data('data/examples/pku_training.train.utf8')
    embedder = GapEmbedder()
    process_dataset(dataset, 'cached/vecs.train.csv', embedder)
    
For the third argument, you can use any instance of an Embedder from embed.py.

We can do the same thing to create a CSV of vector embeddings for a
small development corpus (again, be advised that for the final step, 
there are approximately 1200 sentences to process):

    dataset = read_from_training_data('data/examples/pku_training.dev.utf8')
    process_dataset(dataset, 'cached/vecs.dev.csv', embedder)
 
### Training a classifier for segmentation

To train a neural classifier using these embeddings, do the following:

    from experiment import train_from_csv
    net = train_from_csv('cached/vecs.train.csv', 'cached/vecs.dev.csv')
    
### Segmenting an unsegmented Chinese corpus using the trained classifier

To use the trained classifier to segmented an unsegmented corpus, do
the following:

    from experiment import segment_test_file
    segment_test_file(net, embedder, 'data/bakeoff/testing/pku_test.utf8', 'result.txt')
    
This will produce a segmented version of ```pku_test.utf8``` in ```result.txt```.

### Evaluating the quality of the resulting segmentation

To evaluate the quality of the segmentation in ```result.txt```, run the
following script from a terminal window:

    perl data/bakeoff/scripts/score data/bakeoff/gold/pku_training_words.txt data/bakeoff/gold/pku_test_gold.utf8 result.txt

