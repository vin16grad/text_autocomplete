import torch, torch.nn as nn

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb=256, hidden=512, num_layers=2, drop=0.1, pad_id=0):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb, hidden, num_layers=num_layers, batch_first=True, dropout=drop)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        logits = self.proj(h)
        return logits, None

    @torch.no_grad()
    def generate(self, prefix_ids, max_new=20, eos=None, device="cpu"):
        self.eval()
        x = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new):
            logits, _ = self.forward(x)
            next_id = logits[:, -1].argmax(-1)
            x = torch.cat([x, next_id.unsqueeze(0)], dim=1)
            if eos is not None and int(next_id.item()) == eos:
                break
        return x.squeeze(0).tolist()
