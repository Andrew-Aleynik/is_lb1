from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parent


class Client:
    def __init__(self, endpoint: Path | str | None = None) -> None:
        path = Path(endpoint) if endpoint else ROOT / "endpoint.yaml"
        with path.open(encoding="utf-8") as f:
            ep = yaml.safe_load(f)
        self.base_url = str(ep["base_url"]).rstrip("/")
        self.model = ep["model"]
        self._api = OpenAI(base_url=self.base_url, api_key="none")

    @property
    def server_root(self) -> str:
        u = self.base_url
        return u[: -len("/v1")] if u.endswith("/v1") else u

    def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        *,
        temperature: float = 0,
        ignore_eos: bool = True,
        **params: Any,
    ):
        return self._api.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": ignore_eos},
            **params,
        )


if __name__ == "__main__":
    from measure import timed
    from prompts import make_messages

    c = Client()
    r = timed(c, make_messages(c, 128, "smoke"), max_tokens=16)
    print(
        f"ttft={r['ttft_s']:.3f}s tpot={r['tpot_s']:.4f}s "
        f"prompt={r['prompt_tokens']} out={r['output_tokens']}"
    )
