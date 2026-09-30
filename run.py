#!/usr/bin/env python3
"""
启动口语陪练服务。

这是给"记不住 -m app"准备的入口，等价于：

    .venv/bin/python -m app

用法：
    .venv/bin/python run.py                # 本机，浏览器开 http://127.0.0.1:8000
    .venv/bin/python run.py --ssl          # 手机用（需要证书，见 SETUP.md）
    .venv/bin/python run.py --port 8001    # 换端口

之所以单独放一个文件：正式的入口是 `python -m app`，
但很多人（包括我自己）会习惯性敲 `python server.py`。
原型期的 server.py 已经删除，敲错会得到一句
"can't open file 'server.py'"，看起来像工程坏了。
这里给一个显式入口，并把它指到正确的实现上。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.__main__ import main   # noqa: E402

if __name__ == "__main__":
    main()
