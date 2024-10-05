import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.data = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        encoding = self.tokenizer(self.data[index], return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask


class DistiledGPT2:
    def __init__(self, device = torch.device('cuda')):
        self.device = device
        self.load()
    
    def load(self, model_path = 'distilgpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def unload(self):
        self.model = None
        self.tokenizer = None

    def train(self, data, n_epochs=5):
        self.model.train()

        dataset = PromptDataset(data, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        for epoch in range(n_epochs):
            for i, (input_ids, attention_mask) in tqdm(enumerate(dataloader)):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
            
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        self.model.save_pretrained(f'./distilgpt2-finetuned-epoch{epoch+1}')
        self.tokenizer.save_pretrained(f'./distilgpt2-finetuned-epoch{epoch+1}')


    def predict(self, text):
        self.model.eval()
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        attention_mask = torch.tensor([1] * input_ids.shape[1]).unsqueeze(0).to(self.device)
        output = self.model.generate(input_ids, 
                                    attention_mask = attention_mask, 
                                    max_new_tokens=20,
                                    do_sample=True,
                                    num_beams=3,
                                    early_stopping=True,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    temperature=0.9,
                                    )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)