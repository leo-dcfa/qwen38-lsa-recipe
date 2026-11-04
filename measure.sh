#!/usr/bin/env bash
# One measurement, the same way every time: launch a recipe on this box, wait until it
# answers, check that it is the server it claims to be, run the bench, stop it.
#   ./measure.sh baseline bench 131072 262144 524288
#   ./measure.sh lsa bench 131072 262144 524288
#   ./measure.sh baseline needles 32768 131072
# It refuses to start while another container runs here, because two models can run a GB10
# out of memory and hang it, and it kills the server if free memory falls below 6 GB.
# The log is kept in ~/qwen38-lsa-logs, because stopping the server deletes its container.
set -euo pipefail
cd "$(dirname "$0")"
which="${1:?usage: ./measure.sh baseline|lsa bench <contexts...>}"
task="${2:?usage: ./measure.sh baseline|lsa bench <contexts...>}"
shift 2
recipe="./qwen38-$which.yaml"
[ -f "$recipe" ] || { echo "measure.sh: no $recipe" >&2; exit 1; }
./check-recipes.sh
set -a; . ./pins.env; set +a
PATH="$HOME/.local/bin:$PATH"
cache="${HF_HOME:-$HOME/.cache/huggingface}"
image="qwen38-lsa:sglang-${SGLANG_COMMIT:0:7}-lsa-${LSA_COMMIT:0:7}"
commit="$(git rev-parse --short HEAD)"
git diff --quiet HEAD -- "$recipe" bench.py needles.py pins.env \
  || { echo "measure.sh: commit first, so each result names what ran" >&2; exit 1; }

if [ -n "$(docker ps -q)" ]; then
  echo "measure.sh: free this box first; running now:" >&2
  docker ps --format "  {{.Names}}" >&2
  exit 1
fi

hf download --repo-type dataset "$SHAREGPT_REPO" "$SHAREGPT_FILE" \
  --revision "$SHAREGPT_REVISION" > /dev/null
sparkrun run "$recipe" --hosts localhost --no-follow
trap "sparkrun stop \"$recipe\" --hosts localhost > /dev/null 2>&1 || true" EXIT
server="$(docker ps --filter name=sparkrun_ --format "{{.Names}}" | head -n 1)"
mkdir -p ~/qwen38-lsa-logs
log=~/qwen38-lsa-logs/"$(date +%Y%m%d-%H%M)-$which-$task.log"
docker exec "$server" tail -n +1 -F /tmp/sparkrun_serve.log > "$log" 2> /dev/null &

# The guard runs for the whole measurement: a long prefill is when the KV cache fills.
(
  while sleep 2; do
    avail_kb="$(awk "/^MemAvailable/ {print \$2}" /proc/meminfo)"
    if (( avail_kb < 6 * 1024 * 1024 )); then
      docker kill "$server" > /dev/null 2>&1 || true
      echo "measure.sh: only $((avail_kb / 1024)) MB free, server killed; log: $log" >&2
      exit 0
    fi
  done
) &
guard=$!
trap "kill $guard 2> /dev/null || true; sparkrun stop \"$recipe\" --hosts localhost > /dev/null 2>&1 || true" EXIT

for _ in $(seq 2400); do
  if curl -sf -m 2 127.0.0.1:8000/v1/models > /dev/null; then break; fi
  if [ -z "$(docker ps -q --filter "name=^$server\$")" ]; then
    echo "measure.sh: the server exited; log: $log" >&2
    exit 1
  fi
  sleep 1
done
curl -sf -m 2 127.0.0.1:8000/v1/models > /dev/null \
  || { echo "measure.sh: not ready after 40 minutes; log: $log" >&2; exit 1; }
echo "measure.sh: ready; server log: $log"

# One question with a known answer (3 hours 35 minutes) catches the obvious breakage.
python3 - <<"PY"
import json, urllib.request
body = {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
                      "A train leaves at 09:40 and arrives at 13:15. How long is the trip?"}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
answer = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]
print("measure.sh: smoke answer:", " ".join((answer or "").split())[:300])
PY

# The two lines the plugin prints. "armed" says its hooks registered; only "indexing" says an
# LSA index was built. Without both the server quietly runs full attention, and without this
# check that would be scored as LSA.
sleep 2
armed=0; grep -qF "lsa-lsa: armed" "$log" && armed=1
built=0; grep -qF "lsa-lsa: indexing" "$log" && built=1
echo "measure.sh: armed=$armed indexing=$built"
if [ "$which" = lsa ] && (( armed + built != 2 )); then
  echo "measure.sh: this is not an LSA server; log: $log" >&2
  exit 1
fi
if [ "$which" = baseline ] && (( built != 0 )); then
  echo "measure.sh: this baseline built an LSA index; log: $log" >&2
  exit 1
fi

case "$task" in
  bench)
    docker run --rm --network host \
      --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
      -v "$cache:/cache/huggingface:ro" -v "$PWD:/recipe:ro" -v "$PWD/results:/results" \
      --entrypoint python3 "$image" -I /recipe/bench.py \
        --label "$which" --recipe-commit "$commit" \
        --snapshot "/cache/huggingface/hub/models--${MODEL//\//--}/snapshots/$MODEL_REVISION" \
        --text "/cache/huggingface/hub/datasets--${SHAREGPT_REPO//\//--}/snapshots/$SHAREGPT_REVISION/$SHAREGPT_FILE" \
        --contexts "$@"
    ;;
  needles)
    docker run --rm --network host \
      --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
      -v "$cache:/cache/huggingface:ro" -v "$PWD:/recipe:ro" -v "$PWD/results:/results" \
      --entrypoint python3 "$image" -I /recipe/needles.py \
        --label "$which" --recipe-commit "$commit" \
        --snapshot "/cache/huggingface/hub/models--${MODEL//\//--}/snapshots/$MODEL_REVISION" \
        --text "/cache/huggingface/hub/datasets--${SHAREGPT_REPO//\//--}/snapshots/$SHAREGPT_REVISION/$SHAREGPT_FILE" \
        --contexts "$@"
    ;;
  *)
    echo "measure.sh: unknown task $task" >&2
    exit 1
    ;;
esac
