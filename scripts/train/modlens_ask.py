# -*- coding: utf-8 -*-
"""modlens 视觉识别封装：图片 -> 结构化文本描述（英雄名列表等）。

独立进程调用 modlens（避免与主进程模型冲突），输出 JSON。
用法（被 roster_detect 等调用）：
    python scripts/train/modlens_ask.py <图片路径> "<提示词>" [--out out.json]
"""
import argparse
import base64
import json
import os
import sys
import urllib.request

# modlens 配置（与 ~/.modlens/config.json 一致）
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
API_KEY = "efbb1fd390a148d7a41015d0fed50c88.OA3jiaSzZZQKmLGB"
MODEL = "glm-4.6v"


def ask(image_path, prompt, out=None):
    """调用智谱 glm-4.6v 识图。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    text = data["choices"][0]["message"]["content"]
    if out:
        Path(out).write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    return text


if __name__ == "__main__":
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="图片路径")
    ap.add_argument("prompt", help="提示词")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        t = ask(args.image, args.prompt, args.out)
        print(t)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
