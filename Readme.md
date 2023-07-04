# ContentModeration

Content moderation is an important part of keeping online platforms safe and welcoming for everyone. It is a challenging task, but it is essential for ensuring that online platforms are used for good.

This repository contains a classification system that identifies if the user input is acceptable.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training Scripts

LSVC: ['https://www.kaggle.com/vidmath25/lvc-v2/edit']
LSTM: ['https://colab.research.google.com/drive/1lL7lIpuQlqq4U-tFel4aOuQWmUzrgK50#scrollTo=H0Zpp7NlU2aQ']

## Usage

The code in this repository can be used to deply a profanity detector to classify harmful texts. To do this, you can use the server_LSVC_LSTM.py module. This module takes a text as input and returns a prompt if acceptable or not.

