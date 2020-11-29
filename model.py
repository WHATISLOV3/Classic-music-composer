import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModule(nn.Module):
    def __init__(self, sequence_size, hidden_size,
                embedding_size, batch_size, n_notes):
        super(RNNModule, self).__init__()
        self.seq_size = sequence_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(n_notes, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True)
        self.dense = nn.Linear(hidden_size, n_notes)
        
    
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        return logits, state
    
    
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size), 
                torch.zeros(1, batch_size, self.hidden_size))