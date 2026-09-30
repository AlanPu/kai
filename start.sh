#!/usr/bin/env bash
#
# 启动口语陪练服务。
#
# 等价于：
#     pkill -f run.py
#     .venv/bin/python run.py
#
# 但比这两行多做了几件容易忘、忘了就会困惑的事 ——
# 都是实际踩过的坑：
#
#   1. 先把已在跑的旧进程停掉。
#      不停的话新进程会因为端口被占用而启动失败，但旧进程还在响应，
#      于是"改了代码却没生效" —— 最费解的一类问题。
#
#   2. 等端口真正释放再启动。
#      pkill 发出信号后进程不是立刻消失的。立刻启动会撞上
#      "Address already in use"。
#
#   3. 启动后确认真的起来了。
#      配置错（比如模型名写错、密钥过期）时进程会活着但对话不可用。
#      这里主动请求一次，起不来就直接告诉你。
#
#   4. 日志写到 data/logs/，出问题能回看。
#
# 用法：
#     ./start.sh                  # 本机访问 http://127.0.0.1:8000
#     ./start.sh --ssl            # 手机/平板访问，https（证书已生成过）
#     ./start.sh --port 8001      # 换端口
#     ./start.sh --reload         # 改代码自动重启（开发用）
#     ./start.sh --stop           # 只停服务
#     ./start.sh --status         # 看当前状态
#     ./start.sh --logs           # 跟踪日志
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=".venv/bin/python"
LOG_DIR="data/logs"
LOG="$LOG_DIR/server.log"
PORT=8000

# ---------- 解析出端口，好用来检查服务是否真的起来了 ----------
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == "--port" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    PORT="${ARGS[i + 1]}"
  fi
done

# 找出监听该端口的进程（/bin/ps 在部分环境不可用，用 lsof）
pids_on_port() {
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true
}

stop_server() {
  local pids
  pids="$(pids_on_port)"
  if [[ -z "$pids" ]]; then
    echo "  （端口 $PORT 上没有在跑的服务）"
    return 0
  fi
  echo "  停止旧进程：$(echo "$pids" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  # 等它真的退出，最多 10 秒
  for _ in $(seq 1 20); do
    [[ -z "$(pids_on_port)" ]] && return 0
    sleep 0.5
  done
  echo "  旧进程没退干净，强制结束"
  # shellcheck disable=SC2086
  kill -9 $(pids_on_port) 2>/dev/null || true
  sleep 1
}

case "${1:-}" in
  --stop)
    stop_server
    echo "  ✅ 已停止"
    exit 0
    ;;
  --status)
    pids="$(pids_on_port)"
    if [[ -z "$pids" ]]; then
      echo "  服务未运行（端口 $PORT 空闲）"
      exit 1
    fi
    echo "  运行中，PID：$(echo "$pids" | tr '\n' ' ')"
    printf "  健康检查："
    if curl -s --noproxy '*' --max-time 5 \
        "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
      echo "正常"
    else
      echo "无响应（可能在启动中，或配置有问题 —— 看 ./start.sh --logs）"
    fi
    exit 0
    ;;
  --logs)
    [[ -f "$LOG" ]] || { echo "  还没有日志（$LOG）"; exit 1; }
    tail -f "$LOG"
    exit 0
    ;;
  -h|--help)
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
esac

# ---------- 前置检查 ----------
if [[ ! -x "$PY" ]]; then
  echo "❌ 找不到虚拟环境：$PY"
  echo "   先执行：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "⚠️  没有 .env（从 .env.example 复制一份）"
  echo "   cp .env.example .env"
  exit 1
fi

# --ssl 需要证书；没有就明确说怎么办，而不是让 uvicorn 抛错
for a in "$@"; do
  if [[ "$a" == "--ssl" ]]; then
    if [[ ! -f certs/cert.pem || ! -f certs/key.pem ]]; then
      echo "❌ --ssl 需要证书，但 certs/ 下没有："
      echo "   .venv/bin/python scripts/make_cert.py"
      exit 1
    fi
  fi
done

mkdir -p "$LOG_DIR"

echo "英语口语陪练"
echo "----------"
stop_server

echo "  启动中…"
# -u：不缓冲，日志能实时看到（否则日志文件长时间是空的，像是卡住了）
nohup "$PY" -u run.py "$@" >>"$LOG" 2>&1 &
NEW_PID=$!

# ---------- 等它就绪 ----------
SCHEME="http"
for a in "$@"; do [[ "$a" == "--ssl" ]] && SCHEME="https"; done

READY=0
for _ in $(seq 1 40); do
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo
    echo "❌ 进程启动后立刻退出了。最后几行日志："
    tail -20 "$LOG" | sed 's/^/     /'
    exit 1
  fi
  if curl -s -k --noproxy '*' --max-time 2 \
       "$SCHEME://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 0.5
done

echo
if [[ "$READY" == "1" ]]; then
  # 打印生效的模型/参数，配置错了能立刻发现
  grep -A5 "模型配置" "$LOG" | tail -5 | sed 's/^/  /'
  echo
  echo "  ✅ 已启动  PID $NEW_PID"
  echo "     电脑打开：$SCHEME://127.0.0.1:$PORT"
  if [[ "$SCHEME" == "https" ]]; then
    IP="$(ipconfig getifaddr en0 2>/dev/null \
          || ipconfig getifaddr en1 2>/dev/null || echo '')"
    [[ -n "$IP" ]] && echo "     手机打开：https://$IP:$PORT  （需同一 WiFi）"
  fi
  echo "     看日志：  ./start.sh --logs"
  echo "     停止：    ./start.sh --stop"
else
  echo "⚠️  进程还活着，但 20 秒内没通过健康检查。"
  echo "   可能是配置问题。最后几行日志："
  tail -20 "$LOG" | sed 's/^/     /'
  exit 1
fi
