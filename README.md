# Real-time text sentiment analysis with a custom logistic regression classifier

## Overview

This is a web app demonstraing a real-time sentence-based text sentiment analysis, implemented with a custom-built [logistic regression classifier](https://en.wikipedia.org/wiki/Logistic_regression). The classifier is trained on NLTK sample Twitter data set (see [Twitter Samples](http://www.nltk.org/nltk_data/)).

The project consists of the following modules:
* `text-sentiment-app`: a React application, which captures text input from a user, splits captured text into sentences, sends each sentence which was not classified yet via a websocket to `text-sentiment-server` for classification, and then visualises classified sentences with a sentiment-based color scale.
* `text-sentiment-server`: a Python-based server, which gets sentences from `text-sentiment-app` via a websocket, and sets a sentence sentiment based on a pre-trained logistic regression classifier. 

The following are key 3rd party components used:
* [sentence-splitter](https://www.npmjs.com/package/sentence-splitter) - used in `text-sentiment-app` for splitting captured text into sentences
* [NLTK](https://recordrtc.org/) - used in `text-sentiment-server` to pre-process sentences
* [numpy](https://numpy.org/) - used in `text-sentiment-server` for matrix math

## Model configuration

Sentence classification is implemented with a logistic regression classifier. Classifier is pre-trained based on Twitter samples (10k samples in total, 5k positives and 5k negatives), on a word level. Split factor (training / test data) is 0.8.

`text-sentiment-server` checks on starting-up if a pre-trained model is available - and if not, runs a model pre-training, tests trained model and saves that in binaries - you should see an output similar to this:
```
2020-12-23 15:16:24,333 [MainThread  ] [INFO ]  No pre-trained model available, training now...
2020-12-23 15:16:24,333 [MainThread  ] [INFO ]  Retrieving dataset.
[nltk_data] Downloading package twitter_samples to
[nltk_data]     /Users/olegas.murasko/nltk_data...
[nltk_data]   Package twitter_samples is already up-to-date!
[nltk_data] Downloading package stopwords to
[nltk_data]     /Users/olegas.murasko/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
2020-12-23 15:16:25,338 [MainThread  ] [INFO ]  Splitting dataset, split factor: 0.8
2020-12-23 15:16:28,464 [MainThread  ] [INFO ]  Completed frequency matrix, total length: 11340
2020-12-23 15:16:31,977 [MainThread  ] [INFO ]  The cost after training is 0.24216477.
2020-12-23 15:16:31,978 [MainThread  ] [INFO ]  The resulting vector of weights is [7e-08, 0.0005239, -0.00055517]
2020-12-23 15:16:32,861 [MainThread  ] [INFO ]  Accuracy on test dataset: 0.995
2020-12-23 15:16:32,868 [MainThread  ] [INFO ]  Loaded model.
2020-12-23 15:16:32,868 [MainThread  ] [INFO ]  Starting server on host 0.0.0.0, port 2702
```

Training accuracy can be tuned by changing training iteration count (now set to 1500). 
Trained model (freatures and weights) are stored in binary files aside `text-sentiment-server` runtime (subfolder `model`).

## Setting-up

Clone full project:
```
git clone git@github.com:uphop/text-sentiment.git && cd text-sentiment
```

Install dependencies and prepare configuration for `text-sentiment-app`:
```
cd text-sentiment-app && yarn install && cp .env.sample .env && cd..
```

Install dependencies and prepare configuration for `text-sentiment-server`:
```
cd text-sentiment-server && pip3 install -r requrements.txt && cp .example-env .env && cd..
```

## Starting-up

Start `sentiment-assessment-server`:
```
cd text-sentiment-server && ./run.sh
```

Start `text-sentiment-app`:
```
cd voice-sentiment-app &&  yarn start
```

## Usage

Type some text into the text area - the app will be capturing text, attempting to split into sentences / assess sentiment scores, and visualise those as text background color.

Here is an example of what you should see as the result:
![Screenshot](https://user-images.githubusercontent.com/74451637/103000607-089c2980-4534-11eb-9104-a8bb372ec2ff.png)

And here is a recorded example of sentiment score assessment while typing:
[![Recorded_sample](http://img.youtube.com/vi/0WAfJ4Dj-8U/0.jpg)](http://www.youtube.com/watch?v=0WAfJ4Dj-8U "Text Sentiment example")



