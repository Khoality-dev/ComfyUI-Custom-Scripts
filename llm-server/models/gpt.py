import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=150):
        self.data = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cached = [None] * len(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.cached[index] is not None:
            return self.cached[index][0], self.cached[index][1]
        
        encoding = self.tokenizer(self.data[index], return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        self.cached[index] = (input_ids, attention_mask)
        return input_ids, attention_mask


class GPTModel:
    def __init__(self, model_path = 'distilgpt2', device = torch.device('cuda')):
        self.device = device
        self.load(model_path)
    
    def load(self, model_path = 'distilgpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None

    def train(self, data, n_epochs=5):
        self.model.train()

        dataset = PromptDataset(data, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=5,
            num_training_steps=(len(dataloader) * n_epochs),
        )
        
        
        for epoch in range(n_epochs):
            total_loss = 0
            for input_ids, attention_mask in tqdm(dataloader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                total_loss += loss.item()

            print(f"Epoch: {epoch+1}, Loss: {total_loss/len(dataloader)}")

            if ((epoch+1) % 5 == 0):
                print("Saving model...")
                self.model.save_pretrained(f'./model-{epoch+1}')
                self.tokenizer.save_pretrained(f'./model-{epoch+1}')


    def predict(self, text):
        self.model.eval()
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        attention_mask = torch.tensor([1] * input_ids.shape[1]).unsqueeze(0).to(self.device)
        output = self.model.generate(input_ids, 
                                    attention_mask = attention_mask, 
                                    max_new_tokens=20,
                                    do_sample=True,
                                    num_beams=20,
                                    top_p = 0.9,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    temperature=0.9,
                                    no_repeat_ngram_size=5,
                                    num_return_sequences=15
                                    )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)