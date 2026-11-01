#!/usr/bin/env python3
"""Sinov-Oybek modelini yuklaydi va javob yozadi. Terminalda ishlatish: python chat.py"""
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, Config

ROOT = Path(__file__).resolve().parent
MAX_NEW_TOKENS = 120


class Bot:
    def __init__(self, weights=ROOT / "weights" / "chat_best.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(weights, map_location=self.device, weights_only=True)
        self.model = GPT(Config(**state["config"])).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
        self.user, self.assistant, self.eos = (
            self.tok.token_to_id(t) for t in ("<|user|>", "<|assistant|>", "<|endoftext|>")
        )

    @staticmethod
    def _normalize(text):
        for a, b in (("o'", "o‘"), ("g'", "g‘"), ("O'", "O‘"), ("G'", "G‘")):
            text = text.replace(a, b)
        return text

    def ask(self, text, temperature=0.7, top_k=40):
        ids = self.tok.encode(self._normalize(text), add_special_tokens=False).ids
        room = self.model.config.block_size - MAX_NEW_TOKENS - 2
        prompt = [self.user] + ids[-room:] + [self.assistant]
        x = torch.tensor([prompt], device=self.device)
        out = self.model.generate(x, MAX_NEW_TOKENS, temperature=temperature, top_k=top_k, eos_id=self.eos)
        return self.tok.decode(out[0].tolist()[len(prompt):]).strip()


if __name__ == "__main__":
    bot = Bot()
    print("Sinov-Oybek. Chiqish: Ctrl+C")
    try:
        while True:
            q = input("\nSiz: ").strip()
            if q:
                print("Model:", bot.ask(q))
    except (KeyboardInterrupt, EOFError):
        print()
