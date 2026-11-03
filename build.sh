#!/usr/bin/env bash
# Build the image from the versions in pins.env.
# The tag names the fork and plugin commits, which is what the recipe will refer to.
# A label records this repo commit, so an image can always be traced back to the
# Dockerfile and pins.env it came from.
#   ./build.sh                       extra arguments go to docker build,
#   ./build.sh --progress=plain      e.g. to see every line of output
set -euo pipefail
cd "$(dirname "$0")"

# The label would lie if the files it points to had uncommitted changes.
if ! git diff --quiet HEAD -- Dockerfile pins.env; then
  echo "build.sh: commit Dockerfile and pins.env first" >&2
  exit 1
fi

set -a; . ./pins.env; set +a

tag="qwen38-lsa:sglang-${SGLANG_COMMIT:0:7}-lsa-${LSA_COMMIT:0:7}"
docker build \
  --build-arg SGLANG_REPO --build-arg SGLANG_COMMIT \
  --build-arg PYTHON_VERSION --build-arg EXCLUDE_NEWER \
  --build-arg LSA_REPO --build-arg LSA_COMMIT \
  --build-arg MAP_URL --build-arg MAP_FILE --build-arg MAP_SHA256 \
  --label org.opencontainers.image.source=https://github.com/leo-dcfa/qwen38-lsa-recipe \
  --label org.opencontainers.image.revision="$(git rev-parse HEAD)" \
  -t "$tag" "$@" .
echo "build.sh: built $tag"
