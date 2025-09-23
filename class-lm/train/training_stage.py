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

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
from train.dset import PackedBinDataset
from model.tiny_gpt import TinyGPT
from tokenizers import Tokenizer

DEFAULT_TOKENIZER_PATH = _ROOT / "tokenizer" / "wordpiece" / "tokenizer.json"

def cycle(dl):
    while True:
        for x in dl: yield x


def evaluate_model(
    model: TinyGPT,
    loss_fn: nn.Module,
    dl: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    autocast_enabled: bool,
    vocab_size: int,
    max_batches: int = 0,
) -> tuple[float, float, int]:
    """Run a quick eval pass and return (avg_loss, ppl, batches_evaluated)."""
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        )
        for i, (x, y) in enumerate(dl):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1))
            total_loss += float(loss.item())
            batches += 1
    model.train()
    if batches == 0:
        return float("nan"), float("nan"), 0
    avg = total_loss / batches
    ppl = math.exp(min(avg, 20.0))
    return avg, ppl, batches


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
                probs = F.softmax(next_logits.float(), dim=-1)
                if not torch.isfinite(probs).all():
                    probs = torch.zeros_like(probs)
                    probs[..., 0] = 1.0
                sums = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(sums > 0, probs / sums, torch.full_like(probs, 0.0))
                fallback = (probs.sum(dim=-1) <= 0).any()
                next_token = (
                    torch.argmax(next_logits, dim=-1, keepdim=True)
                    if fallback
                    else torch.multinomial(probs, num_samples=1)
                )
                generated = torch.cat([generated, next_token], dim=1)
                if eos_id is not None and next_token.item() == eos_id:
                    break

            text = tokenizer.decode(generated.squeeze(0).tolist(), skip_special_tokens=True)
            results.append((prompt, text))

    model.train()
    return results

def _emit_samples(
    *,
    tag: str,
    model: TinyGPT,
    tokenizer: Tokenizer,
    sample_prompts: list[str],
    sample_count: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    top_k: int,
):
    prompt_pool = sample_prompts or [""]
    prompts = [prompt_pool[i % len(prompt_pool)] for i in range(sample_count)]
    samples = sample_model(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k,
    )
    for idx, (prompt, text) in enumerate(samples, start=1):
        print(f"[sample {tag}-{idx}] prompt: {prompt!r}\n{text}\n")


def main(a):
    torch.manual_seed(1337)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TinyGPT(
        vocab=a.vocab,
        n_layers=a.layers,
        d_model=a.dmodel,
        n_heads=a.heads,
        d_ff=a.ff,
        max_seq=a.seq,
        dropout=a.dropout,
    ).to(device)
    opt = optim.AdamW(
        model.parameters(),
        lr=a.lr,
        betas=(a.beta1, a.beta2),
        weight_decay=a.weight_decay,
    )
    # Defer scheduler construction until after potential resume so base_lrs match a.lr
    sched = None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=getattr(a, "label_smoothing", 0.0))

    ds = PackedBinDataset(a.bin, seq_len=a.seq)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=a.mb,
        shuffle=a.shuffle,
        num_workers=a.workers,
        persistent_workers=bool(a.workers > 0),
    )
    it = cycle(dl)

    sample_enabled = (a.sample_every > 0) or a.sample_on_finish
    tokenizer = None
    sample_prompts = a.sample_prompt or []
    if sample_enabled:
        tokenizer_path = a.tokenizer_path or DEFAULT_TOKENIZER_PATH
        if not tokenizer_path.exists():
            print(f"[sampling] disabled: tokenizer not found at {tokenizer_path}")
            sample_enabled = False
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            if not sample_prompts:
                sample_prompts = [
                    "Explain dynamic arrays",
                    "What is a heap?",
                ]
    sampling_possible = sample_enabled and tokenizer is not None

    model.train()
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda and not a.no_autocast)
    autocast_enabled = (device.type in {"cuda", "mps"}) and not a.no_autocast
    print(
        f"[config] autocast_enabled={autocast_enabled} "
        f"scaler_enabled={scaler.is_enabled()}"
    )
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
    lr_decayed = False
    start_step = 0
    # Optional validation setup
    val_enabled = bool(a.val_every > 0 and a.val_bin is not None)
    val_dl = None
    val_metrics_path = None
    if val_enabled:
        try:
            val_ds = PackedBinDataset(a.val_bin, seq_len=a.seq)
        except Exception as e:
            print(f"[val] disabled: could not load '{a.val_bin}': {e}")
            val_enabled = False
            val_ds = None
        if val_enabled and val_ds is not None:
            val_mb = a.val_mb if a.val_mb is not None else a.mb
            val_dl = torch.utils.data.DataLoader(
                val_ds,
                batch_size=val_mb,
                shuffle=False,
                num_workers=a.workers,
                persistent_workers=bool(a.workers > 0),
            )
            if a.val_metrics_file is not None:
                val_metrics_path = Path(a.val_metrics_file)
                val_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                if not val_metrics_path.exists():
                    with val_metrics_path.open("w", newline="") as fh:
                        csv.writer(fh).writerow(["step", "val_loss", "val_perplexity"]) 
    if a.resume:
        ckpt = torch.load(a.resume, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        state_dict.pop("alibi_cache", None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[resume] missing keys ignored: {missing}")
        if unexpected:
            print(f"[resume] unexpected keys ignored: {unexpected}")
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            # Ensure the resumed run honours the user-specified LR.
            for group in opt.param_groups:
                group["lr"] = a.lr
                # Some schedulers read initial_lr; keep it in sync as well.
                group["initial_lr"] = a.lr
        if sched is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            try:
                sched.load_state_dict(ckpt["scheduler"])
            except Exception:
                print("[warn] scheduler state incompatible; continuing without loading")
        if use_cuda and "scaler" in ckpt and ckpt["scaler"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                print("[warn] scaler state incompatible; resetting scaler")
        lr_decayed = ckpt.get("lr_decayed", False)
        start_step = ckpt.get("step", 0)
        print(f"[resume] Loaded checkpoint '{a.resume}' at step {start_step}")

    # Build scheduler after resume so base_lrs reflect the desired --lr and remaining steps
    remaining_steps = max(0, a.steps - start_step)
    warmup_steps = min(max(0, a.warmup_steps), remaining_steps)
    total_main_steps = max(1, remaining_steps - warmup_steps)
    if a.lr_scheduler == "cosine":
        main_sched = CosineAnnealingLR(opt, T_max=total_main_steps, eta_min=a.lr_eta_min)
    elif a.lr_scheduler == "step":
        main_sched = StepLR(opt, step_size=a.lr_scheduler_step_size, gamma=a.lr_scheduler_gamma)
    elif a.lr_scheduler == "poly":
        def poly_lambda(step: int) -> float:
            progress = min(step, total_main_steps) / max(1, total_main_steps)
            return max((1.0 - progress) ** a.lr_scheduler_power, 0.0)

        main_sched = LambdaLR(opt, lr_lambda=poly_lambda)
    else:
        main_sched = LambdaLR(opt, lr_lambda=lambda step: 1.0)

    if warmup_steps > 0:
        warmup = LinearLR(
            opt,
            start_factor=a.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        sched = SequentialLR(opt, schedulers=[warmup, main_sched], milestones=[warmup_steps])
    else:
        sched = main_sched

    # One-time LR debug to verify warmup/base LR
    try:
        base_lr = opt.param_groups[0]["lr"]
        print(
            f"[lr-setup] base_lr={a.lr:.2e} warmup_steps={warmup_steps} "
            f"start_factor={a.warmup_start_factor} current_lr={base_lr:.2e}"
        )
    except Exception:
        pass
    for step in range(start_step + 1, a.steps + 1):
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
            # print(
            #     f"[logits] step={step} micro={_} stats=",
            #     dict(
            #         min=float(logits.min()),
            #         max=float(logits.max()),
            #         mean=float(logits.mean()),
            #         std=float(logits.std()),
            #     ),
            # )
            if step == 1 and _ == 0:
                print(f"debug raw loss {raw_loss.item():.6f} | scaled {loss.item():.6f}")
            if not torch.isfinite(raw_loss):
                print(
                    f"[nan-debug] step={step} micro={_} raw={raw_loss.item()} scaled={loss.item()} "
                    f"lr={opt.param_groups[0]['lr']:.3e}"
                )
                print(
                    "[nan-debug] logits stats:",
                    dict(
                        min=float(logits.min()),
                        max=float(logits.max()),
                        mean=float(logits.mean()),
                        std=float(logits.std()),
                    ),
                )
                print(
                    "[nan-debug] target stats:",
                    dict(min=int(y.min()), max=int(y.max())),
                )
                # continue the loop so gradients don't explode this step
                continue
            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += loss.item()
        if use_cuda:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        if use_cuda:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if a.lr_decay_step and not lr_decayed and step >= a.lr_decay_step:
            for group in opt.param_groups:
                group['lr'] *= a.lr_decay_factor
            lr_decayed = True
            print(f"[lr] decayed to {opt.param_groups[0]['lr']:.2e} at step {step}")

        if sched is not None:
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

        # Periodic validation
        if val_enabled and (step % a.val_every == 0):
            v_loss, v_ppl, v_batches = evaluate_model(
                model,
                loss_fn,
                val_dl,
                device=device,
                autocast_enabled=autocast_enabled,
                vocab_size=a.vocab,
                max_batches=max(0, a.val_max_batches),
            )
            if v_batches > 0:
                print(f"val step {step} | loss {v_loss:.3f} | ppl {v_ppl:.1f} | batches {v_batches}")
                if val_metrics_path is not None:
                    with val_metrics_path.open("a", newline="") as fh:
                        csv.writer(fh).writerow([step, v_loss, v_ppl])

        if sampling_possible and a.sample_every > 0 and step % a.sample_every == 0:
            _emit_samples(
                tag=f"step{step}",
                model=model,
                tokenizer=tokenizer,
                sample_prompts=sample_prompts,
                sample_count=a.sample_count,
                max_new_tokens=a.sample_max_new,
                device=device,
                temperature=a.sample_temperature,
                top_k=a.sample_top_k,
            )

        if step % a.ckpt == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict() if sched is not None else None,
                    "scaler": scaler.state_dict() if use_cuda else None,
                    "cfg": vars(a),
                    "step": step,
                    "lr_decayed": lr_decayed,
                },
                f"{a.out}/ckpt_{step}.pt",
            )

    metrics_fh.close()
    print(f"Saved metrics to {metrics_path}")
    if sampling_possible and a.sample_on_finish:
        _emit_samples(
            tag="final",
            model=model,
            tokenizer=tokenizer,
            sample_prompts=sample_prompts,
            sample_count=a.sample_count,
            max_new_tokens=a.sample_max_new,
            device=device,
            temperature=a.sample_temperature,
            top_k=a.sample_top_k,
        )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True, help="path to .bin (generic for Stage A)")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--dmodel", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2, help="Model dropout probability")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--mb", type=int, default=16)           # micro-batch
    p.add_argument("--accum", type=int, default=8)       # grad accumulation
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    p.add_argument("--beta2", type=float, default=0.99, help="Adam beta2")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    p.add_argument("--clip", type=float, default=0.5, help="Gradient norm clip (L2)")
    p.add_argument("--lr-scheduler", choices=["cosine", "step", "poly", "none"], default="cosine",
                   help="Which LR scheduler to use")
    p.add_argument("--lr-eta-min", type=float, default=1e-5, help="Cosine scheduler LR floor (eta_min)")
    p.add_argument("--lr-scheduler-step-size", type=int, default=1000,
                   help="StepLR: number of steps between decays")
    p.add_argument("--lr-scheduler-gamma", type=float, default=0.5,
                   help="StepLR: multiplicative factor of learning rate decay")
    p.add_argument("--lr-scheduler-power", type=float, default=16.0,
                   help="Poly scheduler: decay exponent (higher = more aggressive early drop)")
    p.add_argument("--steps", type=int, default=6000)
    # Warmup options
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps (0 disables)")
    p.add_argument(
        "--warmup-start-factor",
        type=float,
        default=0.01,
        help="Initial LR factor during warmup (e.g., 0.01 starts at 1% of base LR)",
    )
    p.add_argument("--log", type=int, default=15, help="Log every N steps (0 to disable)")
    p.add_argument("--log-secs", type=float, default=0.0, help="Log every N seconds")
    p.add_argument("--ckpt", type=int, default=500)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True, help="Shuffle training DataLoader")
    p.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers for train/val")
    p.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="tokenizer.json for sampling (default: tokenizer/wordpiece/tokenizer.json)",
    )
    p.add_argument("--sample-every", type=int, default=0, help="Generate samples every N steps (0 to disable)")
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--sample-max-new", type=int, default=80)
    p.add_argument("--sample-prompt", action="append", default=[])
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--sample-top-k", type=int, default=0)
    p.add_argument(
        "--sample-on-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit a sample batch when training completes (default: enabled)",
    )
    p.add_argument("--metrics-file", type=Path, help="Optional CSV destination for logged metrics")
    p.add_argument("--lr-decay-step", type=int, default=0, help="Step at which to decay LR (0 disables)")
    p.add_argument("--lr-decay-factor", type=float, default=0.5, help="Factor to multiply LR when decaying")
    p.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    p.add_argument("--no-autocast", action="store_true", help="Disable autocast (use full precision)")
    # Validation options
    p.add_argument("--val-bin", type=Path, help="Optional validation .bin for periodic eval")
    p.add_argument("--val-every", type=int, default=0, help="Run validation every N steps (0 disables)")
    p.add_argument("--val-mb", type=int, help="Validation micro-batch size (defaults to --mb)")
    p.add_argument("--val-max-batches", type=int, default=50, help="Max batches per validation run (0 = full val set)")
    p.add_argument("--val-metrics-file", type=Path, help="Optional CSV for validation metrics")
    # Loss options
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy label smoothing (0-1)")
    a = p.parse_args()
    import os; os.makedirs(a.out, exist_ok=True)
    main(a)
