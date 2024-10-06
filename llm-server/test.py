import os
from models.gpt import DistiledGPT2
import time

current_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    model = DistiledGPT2(os.path.join(current_dir, '../distilgpt2-finetuned'), device='cuda')

    while True:
        input_string = input("Enter text: ")
        start_time = time.time()
        output = model.predict(input_string)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(output)

if __name__ == '__main__':
    main()