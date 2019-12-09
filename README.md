# INLP_neural_practical_session





This repository aims to deal with a text classification example using either classical ML methods or deep neural models.
We first start with simple models and then try to do some feature engineering to improve the performance on a sample text classification problem. We then investigate deep neural models using e.g. recurrent neural networks to show how powerful they can be with minimal feature engineering.

## Setup
The code is tested on python 3.7. We use Jupyter notebooks so that it becomes easier for students to test different training scenarios without loading everything from scratch.
We also recommend creating a virtual environment (e.g. using Anaconda) to install all the needed libraries/packages. You might find the following links useful for installing Anaconda and also PyTorch (which is needed in this repo for training some neural models):  
<https://www.anaconda.com/download/>   
<https://anaconda.org/pytorch/pytorch>  
You can use ``` preparation_notebook.ipynb ``` notebook to make sure that you have all the needed libraries/datasets for this practical session.
### Needed libraries
After cloning repo, you can install needed libraries using pip command:
``` 
pip install -r requirement.txt
``` 
## Down-stream task
At first, we focus on 20 news-group dataset for text-classification using classic learning algorithms to show their power, especially when dealing with long doucment samplels. This dataset comprises around 19K news posts on 20 topics.  
To demonstrate the power of neural networks without using a very complex architecture, we perform sentiment analysis on IMDB reviews dataset, using LSTM layer that considers order in input sentences (unlike e.g. TFIDF which completely ignores word order in the input text)
### Standard ML approach
We first use simple feature extractors with classical ML classifiers, and then try to improve the performance step-by-step, either by enriching the feature extractor, or by changing the classifier which can be found in ``` Notebook_Classic.ipynb ``` notebook. (You can open a notebook by first using ``` jupyter notebook ``` command and then navigating to the desired notebook).  
### Neural approach
This repo also includes a separate notebook for more advanced ML methods and we aim to classify IMDB reviews based on their sentiment. Here, we don't intend to use advanced architectures in this introductory session; instead, we try to demonstrate how easy it is to implement your desired architecture with libraries like PyTorch. Moreover, we can see that without specific feature engineering for the down-stream task, deep models are able to learn useful complex representations which can help us to e.g. separate different classes of data points in the final document embedding space more easily. These approaches are implemented in ``` Notebook_neural.ipynb ``` notebook.
### Classification with fastText library
We also present a straight-forward method that can give us a baseline for text classification in a very short amount of time. fastText library offers a supervised text classification module that uses Bag of Words (BoW) feature (simple average of embeddings), which as shown in their paper (<https://arxiv.org/pdf/1607.01759.pdf>) gives quite reasonable performance compared to more complex models. We also do some experiments with fastText in ``` Notebook_fastText.ipynb ``` notebook, which we again aim to enhance the performance by changing different hyper parameters of the model input.  
fastText installation guide can be found in their github page: <https://github.com/facebookresearch/fastText>
