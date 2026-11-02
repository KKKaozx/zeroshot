"""Frozen CLIP worker in existing project environment, on inherited private FD."""
import argparse
import json
from pathlib import Path
import socket
import numpy as np
from wire import receive, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--assets", required=True)
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    sock.settimeout(120)
    try:
        import torch
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
        from resource_probe import state_hash
        contract = json.loads(Path(__file__).with_name("live_contract.json").read_text())
        assert torch.cuda.is_available()
        model = CLIPModel.from_pretrained(args.assets, local_files_only=True)
        model.requires_grad_(False).eval()
        before = state_hash(model)
        processor = CLIPImageProcessor.from_pretrained(args.assets, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(args.assets, local_files_only=True)
        text = tokenizer(contract["fixed_text"], return_tensors="pt", truncation=True, max_length=77)
        model.cuda()
        with torch.inference_mode():
            text_tokens = model.text_model(input_ids=text["input_ids"].cuda(),
                attention_mask=text["attention_mask"].cuda()).last_hidden_state.cpu().numpy()
        send(sock, {"status": "ready", "all_frozen": True},
             {"text": text_tokens, "text_mask": text["attention_mask"].numpy()})
        requests = 0
        while True:
            metadata, arrays = receive(sock)
            request_id = metadata["request_id"]
            if metadata["operation"] == "close":
                unchanged = state_hash(model) == before
                assert unchanged and all(not p.requires_grad and p.grad is None for p in model.parameters())
                send(sock, {"request_id": request_id, "status": "closed", "requests": requests,
                            "weights_unchanged": unchanged, "all_frozen": True})
                break
            if metadata["operation"] != "encode":
                raise ValueError("Unsupported operation")
            raw = arrays["raw_images"]
            if (raw.dtype != np.float32 or raw.ndim != 4 or raw.shape[1:] != (96, 96, 3)
                    or not 1 <= len(raw) <= 32 or not np.isfinite(raw).all()
                    or raw.min() < 0 or raw.max() > 255):
                raise ValueError("Unexpected raw image payload")
            tokens = []
            with torch.inference_mode():
                for start in range(0, len(raw), 8):
                    part = raw[start:start + 8]
                    count = len(part)
                    if count < 8:
                        part = np.concatenate((part, np.repeat(part[-1:], 8 - count, axis=0)))
                    pixels = processor(images=list(part), return_tensors="pt", do_rescale=True)["pixel_values"].cuda()
                    tokens.append(model.vision_model(pixel_values=pixels).last_hidden_state[:count].cpu().numpy())
            requests += 1
            send(sock, {"request_id": request_id, "status": "ok"}, {"visual": np.concatenate(tokens)})
    except Exception as exc:
        try:
            send(sock, {"status": "error", "error": type(exc).__name__ + ": " + str(exc)})
        finally:
            sock.close()
        raise
    finally:
        sock.close()


if __name__ == "__main__":
    main()
