import json
import os
import argparse
from models.n_grams import *
from models.distiled_gpt2 import *

def main():
    with open(os.path.join(os.path.dirname(__file__), 'prompts.json')) as f:
        dataset = json.load(f)

    texts = [image_data for image_data in dataset['data']]
    texts = [x for text in texts for x in text['prompt']]

    device = torch.device('cuda')
    model = DistiledGPT2(device)
    model.train(texts, n_epochs= 10)


if __name__ == '__main__':
    main()
