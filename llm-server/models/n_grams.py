from typing import List


class NGrams:
    def __init__(self, data):
        self.data = data
        _, self.bi_grams_counts = self.__data_to_counts(2)
        self.courpus, self.uni_grams_counts = self.__data_to_counts(1)

    def predict(self, x, n = 1):
        tokens = [token.strip() for token in x.split(",")]
        
        best_probabilities = {}

        for end_token in self.courpus:
            best_probabilities[end_token] = (x + "," + end_token, self.bi_grams_counts.get(tokens[-1] +","+ end_token, 0) / self.uni_grams_counts.get(tokens[-1], 0))
    
        for i in range(n):
            new_best_probabilities = {}
            for end_token in self.courpus:
                for prior_token in self.courpus:
                    bi_gram_prob = self.bi_grams_counts.get(prior_token +","+ end_token, 0) / self.uni_grams_counts.get(prior_token, 0)
                    if new_best_probabilities.get(end_token, ("",0))[1] < best_probabilities.get(prior_token, ("",0))[1] * bi_gram_prob:
                        new_best_probabilities[end_token] = (best_probabilities.get(prior_token, 0)[0] + "," + end_token, best_probabilities.get(prior_token, 0)[1] * bi_gram_prob)
            best_probabilities = new_best_probabilities

        best_end_token = max(best_probabilities, key=lambda x: best_probabilities[x][1])
        prediction = best_probabilities[best_end_token][0]
        return prediction

    def __data_to_counts(self, n):
        c = {}
        for data in self.data['data']:
            for prompt in data['prompt']:
                tokens = [token.strip() for token in prompt.split(",")]
                for i in range(len(tokens) - n + 1):
                    n_gram = ",".join(tokens[i:i+n])
                    if n_gram in c:
                        c[n_gram] += 1
                    else:
                        c[n_gram] = 1
        corpus = list(c.keys())
        return corpus, c

