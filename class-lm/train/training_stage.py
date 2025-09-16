import math, argparse, csv, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from contextlib import nullcontext
from time import time
from pathlib import Path
import sys

# Allow running as a script or via `python -m class-lm.train.training_stage`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from torch.optim.lr_scheduler import CosineAnnealingLR
from train.dset import PackedBinDataset
from model.tiny_gpt import TinyGPT
from tokenizers import Tokenizer

def cycle(dl):
    while True:
        for x in dl: yield x


def sample_model(
    model: TinyGPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    top_k: int,
) -> list[tuple[str, str]]:
    model.eval()
    results: list[tuple[str, str]] = []
    eos_id = tokenizer.token_to_id("[SEP]")
    max_seq = getattr(model, "max_seq", 512)

    with torch.no_grad():
        for prompt in prompts:
            encoding = tokenizer.encode(prompt)
            input_ids = torch.tensor(encoding.ids, dtype=torch.long, device=device).unsqueeze(0)
            generated = input_ids

            for _ in range(max_new_tokens):
                if generated.size(1) > max_seq:
                    generated = generated[:, -max_seq:]
                logits = model(generated)
                next_logits = logits[:, -1, :]
                if temperature != 1.0:
                    next_logits = next_logits / max(temperature, 1e-6)
                if top_k > 0:
                    values, _ = torch.topk(next_logits, top_k)
                    threshold = values[:, -1].unsqueeze(-1)
                    next_logits = torch.where(next_logits < threshold, torch.full_like(next_logits, -float("inf")), next_logits)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_id is not None and next_token.item() == eos_id:
                    break

            text = tokenizer.decode(generated.squeeze(0).tolist(), skip_special_tokens=True)
            results.append((prompt, text))

    model.train()
    return results

def main(a):
    torch.manual_seed(1337)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TinyGPT(vocab=a.vocab, n_layers=a.layers, d_model=a.dmodel, n_heads=a.heads,
                    d_ff=a.ff, max_seq=a.seq, dropout=0.1).to(device)
    opt = optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9,0.95), weight_decay=0.1)
    sched = CosineAnnealingLR(opt, T_max=a.steps)
    loss_fn = nn.CrossEntropyLoss()

    ds = PackedBinDataset(a.bin, seq_len=a.seq)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.mb, num_workers=0)
    it = cycle(dl)

    sample_enabled = a.sample_every > 0
    tokenizer = None
    sample_prompts = a.sample_prompt or []
    if sample_enabled:
        if not a.tokenizer_path:
            print("[sampling] disabled: provide --tokenizer-path")
            sample_enabled = False
        else:
            tokenizer = Tokenizer.from_file(str(a.tokenizer_path))
            if not sample_prompts:
                sample_prompts = [
                    "Explain dynamic arrays",
                    "What is a heap?",
                ]

    model.train()
    use_cuda = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    autocast_enabled = device.type in {"cuda", "mps"}
    accum = a.accum
    tokens_per_step = a.mb * a.seq * accum
    last_log_time = time()
    log_tokens = 0
    log_loss = 0.0
    log_steps = 0
    total_tokens = 0
    metrics_path = Path(a.metrics_file) if a.metrics_file else Path(a.out) / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    init_metrics = not metrics_path.exists()
    metrics_fh = metrics_path.open("a", newline="")
    metrics_writer = csv.writer(metrics_fh)
    if init_metrics:
        metrics_writer.writerow(["step", "loss", "perplexity", "tokens_per_sec", "lr"])
    for step in range(1, a.steps+1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            x,y = next(it)
            x,y = x.to(device), y.to(device)
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.float16)
                if autocast_enabled
                else nullcontext()
            )
            with autocast_ctx:
                logits = model(x)
                raw_loss = loss_fn(logits.view(-1, a.vocab).float(), y.view(-1))
                loss = raw_loss / accum
            if step == 1 and _ == 0:
                print(f"debug raw loss {raw_loss.item():.6f} | scaled {loss.item():.6f}")
            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += loss.item()
        if use_cuda:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_cuda:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()

        step_loss = total / accum
        log_loss += step_loss
        log_tokens += tokens_per_step
        total_tokens += tokens_per_step
        log_steps += 1

        now = time()
        should_log = False
        if a.log > 0 and step % a.log == 0:
            should_log = True
        if a.log_secs > 0 and (now - last_log_time) >= a.log_secs:
            should_log = True

        if should_log:
            elapsed = now - last_log_time if now > last_log_time else 1.0
            tok_per_sec = log_tokens / elapsed
            avg_loss = log_loss / max(1, log_steps)
            ppl = math.exp(min(avg_loss, 20.0))
            progress = step / a.steps
            lr = sched.get_last_lr()[0]
            print(
                f"step {step}/{a.steps} ({progress:5.1%}) | loss {avg_loss:.3f} | "
                f"ppl {ppl:.1f} | tok/s {tok_per_sec:,.0f} | lr {lr:.2e}"
            )
            metrics_writer.writerow([step, avg_loss, ppl, tok_per_sec, lr])
            metrics_fh.flush()
            last_log_time = now
            log_tokens = 0
            log_loss = 0.0
            log_steps = 0

        if sample_enabled and step % a.sample_every == 0:
            prompt_pool = sample_prompts or [""]
            prompts = [prompt_pool[i % len(prompt_pool)] for i in range(a.sample_count)]
            samples = sample_model(
                model,
                tokenizer,
                prompts,
                max_new_tokens=a.sample_max_new,
                device=device,
                temperature=a.sample_temperature,
                top_k=a.sample_top_k,
            )
            for idx, (prompt, text) in enumerate(samples, start=1):
                print(f"[sample {idx}] prompt: {prompt!r}\n{text}\n")

        if step % a.ckpt == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(a)}, f"{a.out}/ckpt_{step}.pt")

    metrics_fh.close()
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True, help="path to .bin (generic for Stage A)")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--dmodel", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff", type=int, default=2048)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--mb", type=int, default=2)           # micro-batch
    p.add_argument("--accum", type=int, default=32)       # grad accumulation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--log", type=int, default=0, help="Log every N steps (0 to disable)")
    p.add_argument("--log-secs", type=float, default=10.0, help="Log every N seconds")
    p.add_argument("--ckpt", type=int, default=2000)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--tokenizer-path", type=Path, help="tokenizer.json for sampling")
    p.add_argument("--sample-every", type=int, default=0, help="Generate samples every N steps (0 to disable)")
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--sample-max-new", type=int, default=80)
    p.add_argument("--sample-prompt", action="append", default=[])
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--sample-top-k", type=int, default=50)
    p.add_argument("--metrics-file", type=Path, help="Optional CSV destination for logged metrics")
    a = p.parse_args()
    import os; os.makedirs(a.out, exist_ok=True)
    main(a)
