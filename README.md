# Text Classification

PyTorch re-implementation of some text classificaiton models.

## Introduction
There are a number of discrete stages in the process of Text Classification:

- Data preparation: This involves collecting and preparing the data for training. This may involve tasks such as data cleaning, preprocessing, and splitting the data into training and testing sets.

- Feature extraction: The text data needs to be transformed into a numerical format that can be used as input for the model. This may involve techniques such as bag-of-words, tf-idf, or word embeddings.

- Model selection: Choose an appropriate model architecture that can learn from the extracted features and make predictions. Common models used in text classification include logistic regression, support vector machines, and deep learning models such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

- Training the model: The model is trained on the labeled training data by optimizing a loss function to minimize the error between the predicted outputs and the true labels. This involves iterative updates to the model's parameters using a training algorithm such as gradient descent.

- Model evaluation: The trained model is evaluated on the test set to measure its performance. Metrics such as accuracy, precision, recall, and F1 score are commonly used to evaluate the performance of text classification models.

- Model tuning: Based on the performance of the model on the test set, the model's hyperparameters and architecture may be tuned to improve its performance.

- Model deployment: Once the model has been trained and evaluated, it can be used to make predictions on new, unseen text data.



## Supported Models

Train the following models by editing `model_name` item in config files ([here](https://github.com/Renovamen/Text-Classification/tree/master/configs) are some example config files). Click the link of each for details.

- [**Hierarchical Attention Networks (HAN)**](https://github.com/Renovamen/Text-Classification/tree/master/models/HAN) (`han`)

    **Hierarchical Attention Networks for Document Classification.** *Zichao Yang, et al.* NAACL 2016. [[Paper]](https://www.aclweb.org/anthology/N16-1174.pdf)

- [**fastText**](https://github.com/Renovamen/Text-Classification/tree/master/models/fastText) (`fasttext`)

    **Bag of Tricks for Efficient Text Classification.** *Armand Joulin, et al.* EACL 2017. [[Paper]](https://www.aclweb.org/anthology/E17-2068.pdf) [[Code]](https://github.com/facebookresearch/fastText)

- [**Bi-LSTM + Attention**](https://github.com/Renovamen/Text-Classification/tree/master/models/AttBiLSTM) (`attbilstm`)

    **Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.** *Peng Zhou, et al.* ACL 2016. [[Paper]](https://www.aclweb.org/anthology/P16-2034.pdf)

- [**TextCNN**](https://github.com/Renovamen/Text-Classification/tree/master/models/TextCNN) (`textcnn`)

    **Convolutional Neural Networks for Sentence Classification.** *Yoon Kim.* EMNLP 2014. [[Paper]](https://www.aclweb.org/anthology/D14-1181.pdf) [[Code]](https://github.com/yoonkim/CNN_sentence)

- [**Transformer**](https://github.com/Renovamen/Text-Classification/tree/master/models/Transformer) (`transformer`)

    **Attention Is All You Need.** *Ashish Vaswani, et al.* NIPS 2017. [[Paper]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[Code]](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)


## Requirements

First, make sure your environment is installed with:

- Python >= 3.5

Then install requirements:



To install Cuda: Use the link [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

To install torch with cuda support run the following command:

```bash
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

One can test Cuda support with:

```powershell
nvcc --version
```
This will provide details of the version of Cuda you are running.

```python
import torch
torch.cuda.is_available()
```

And additional requirements:

```bash
pip install -r requirements.txt
```


## Dataset

Currently, the following datasets proposed in [this paper](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) are supported:

- AG News   
- DBpedia
- Yelp Review Polarity
- Yelp Review Full
- Yahoo Answers
- Amazon Review Full
- Amazon Review Polarity

All of them can be download [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) (Google Drive). Click [here](notes/datasets.md) for details of these datasets.

You should download and unzip them first, then set their path (`dataset_path`) in your config files. If you would like to use other datasets, they may have to be stored in the same format as the above mentioned datasets.


&nbsp;

## Pre-trained Word Embeddings

Pre-trained word embeddings are pre-calculated feature vectors of words, generated by training a neural network on a large corpus of text. These feature vectors capture the semantic and syntactic relationships between words, and are used to represent words numerically. In the context of text classification, pre-trained word embeddings can be used as input to a neural network model, which learns to classify text data based on these embedded representations of the input text.

Using pre-trained word embeddings can be advantageous in text classification tasks because they allow the neural network to leverage existing knowledge of the relationships between words, which can lead to better performance than training embeddings from scratch on a smaller dataset. Additionally, pre-trained embeddings can reduce the computational resources required to train a neural network, as the embeddings are fixed and do not need to be trained along with the model. Popular pre-trained word embedding models include Word2Vec, GloVe, and FastText.
If you would like to use pre-trained word embeddings (like [GloVe](https://github.com/stanfordnlp/GloVe)), just set `emb_pretrain` to `True` and specify the path to pre-trained vectors (`emb_folder` and `emb_filename`) in your config files. You could also choose to fine-tune word embeddings or not with by editing `fine_tune_embeddings` item.

Or if you want to randomly initialize the embedding layer's weights, set `emb_pretrain` to `False` and specify the embedding size (`embed_size`).


&nbsp;

## Preprocess

Although [torchtext](https://github.com/pytorch/text) can be used to preprocess data easily, it loads all data in one go and occupies too much memory and slows down the training speed, expecially when the dataset is big. 

Therefore, here I preprocess the data manually and store them locally first (where `configs/test.yaml` is the path to your config file):

```bash
python preprocess.py --config configs/example.yaml 
```

For example
We can download the ag_news data set from [here](https://drive.google.com/file/d/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms/view?resourcekey=0-Q5sv-6rQnLTJArwcASJJow) and the word vector set from [here - glove.6B.zip](https://nlp.stanford.edu/projects/glove/) and then run the command:

```bash
python preprocess.py --config .\configs\ag_news\textcnn.yaml
```



Then I load data dynamically using PyTorch's Dataloader when training (see [`datasets/dataloader.py`](datasets/dataloader.py)).

The preprocessing including encoding and padding sentences and building word2ix map. This may takes a little time, but in this way, the training can occupy less memory (which means we can have a large batch size) and take less time. For example, I need 4.6 minutes (on RTX 2080 Ti) to train a fastText model on Yahoo Answers dataset for an epoch using torchtext, but only 41 seconds using Dataloader.

[`torchtext.py`](https://github.com/Renovamen/Text-Classification/blob/abandoned/datasets/torchtext.py) is the script for loading data via torchtext, you can try it if you have interests.


&nbsp;

## Train

To train a model, just run:

```bash
python train.py --config configs/example.yaml
```


For example:

```bash
python train.py --config .\configs\ag_news\textcnn.yaml
```

If you have enabled the tensorboard (`tensorboard: True` in config files), you can visualize the losses and accuracies during training by:

```bash
tensorboard --logdir=<your_log_dir>
```

&nbsp;

## Test

A checkpoint is a saved version of a trained model during the training process. It allows you to save the weights and other parameters of the model at a particular iteration or epoch, so that you can resume training from that point if necessary, or use the saved model for prediction or inference tasks.

We can test a checkpoint and compute accuracy on test set :

```bash
python test.py --config configs/example.yaml
```

For example:

```bash
python test.py --config .\configs\ag_news\textcnn.yaml
```



## Classify

To predict the category for a specific sentence:

First edit the following items in [`classify.py`](classify.py):

```python
checkpoint_path = 'str: path_to_your_checkpoint'

# pad limits
# only makes sense when model_name == 'han'
sentence_limit_per_doc = 15
word_limit_per_sentence = 20
# only makes sense when model_name != 'han'
word_limit = 200
```

Then, run:

```bash
python classify.py
```


&nbsp;

## Performance

Here I report the test accuracy (%) and training time per epoch (on RTX 2080 Ti) of each model on various datasets. Model parameters are not carefully tuned, so better performance can be achieved by some parameter tuning.

|                            Model                             |  AG News   |   DBpedia   | Yahoo Answers |
| :----------------------------------------------------------: | :--------: | :---------: | :-----------: |
| [Hierarchical Attention Network](https://github.com/Renovamen/Text-Classification/tree/master/models/HAN) | 92.7 (45s) | 98.2 (70s)  |  74.5 (2.7m)  |
| [fastText](https://github.com/Renovamen/Text-Classification/tree/master/models/fastText) | 91.6 (8s)  | 97.9 (25s)  |  66.7 (41s)   |
| [Bi-LSTM + Attention ](https://github.com/Renovamen/Text-Classification/tree/master/models/AttBiLSTM) | 92.0 (50s) | 99.0 (105s) |  73.5 (3.4m)  |
| [TextCNN ](https://github.com/Renovamen/Text-Classification/tree/master/models/TextCNN) | 92.2 (24s) | 98.5 (100s) |   72.8 (4m)   |
| [Transformer](https://github.com/Renovamen/Text-Classification/tree/master/models/Transformer) | 92.2 (60s) | 98.6 (8.2m) |  72.5 (14.5m)  |


&nbsp;

## Notes

- The `load_embeddings` method (in [`utils/embedding.py`](utils/embedding.py)) would try to create a cache for loaded embeddings under folder `dataset_output_path`. This dramatically speeds up the loading time the next time.
- Only the encoder part of Transformer is used.


&nbsp;

## License

[MIT](LICENSE)


&nbsp;

## Acknowledgement

This project is based on [sgrvinod/a-PyTorch-Tutorial-to-Text-Classification](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification).
