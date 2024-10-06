import json
import os
import argparse
from models.n_grams import *
from models.gpt import *

def main():
    with open(os.path.join(os.path.dirname(__file__), 'prompts.json')) as f:
        dataset = json.load(f)

    data = dataset['data']

    device = torch.device('cuda')
    model = GPTModel('gpt2', device=device)
    model.train(data, n_epochs= 20)


if __name__ == '__main__':
    main()
