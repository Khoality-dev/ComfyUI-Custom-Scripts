import os
from models.gpt import GPTModel
import time

current_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    model = GPTModel(os.path.join(current_dir, '../model-20'), device='cuda')

    while True:
        input_string = input("Enter text: ")
        start_time = time.time()
        output = model.predict(input_string)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(output)

if __name__ == '__main__':
    main()