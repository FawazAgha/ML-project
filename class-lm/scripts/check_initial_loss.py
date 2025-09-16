#!/usr/bin/env python3
import torch
from class_lm.train.dset import PackedBinDataset
from class_lm.model.tiny_gpt import TinyGPT


def main() -> None:
    ds = PackedBinDataset('class-lm/data/bins/domain_generic.bin', seq_len=128)
    x, y = ds[0]
    model = TinyGPT(vocab=16000, n_layers=2, d_model=128, n_heads=4, d_ff=512, max_seq=128)
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0)).reshape(-1, 16000)
        loss = torch.nn.functional.cross_entropy(logits.float(), y.long())
    print('initial loss', loss.item())


if __name__ == '__main__':
    main()
