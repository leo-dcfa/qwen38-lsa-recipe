#!/usr/bin/env bash
# A recipe is YAML and cannot read pins.env, so it holds copies. Stop if one has drifted.
# The two recipes must also differ only where LSA is armed, or the comparison measures more
# than the read.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ./pins.env; set +a
fail=0
need() { grep -qF -- "$2" "$1" || { echo "check-recipes.sh: $1 lacks $3" >&2; fail=1; }; }
for recipe in qwen38-baseline.yaml qwen38-lsa.yaml; do
  [ -f "$recipe" ] || continue
  need "$recipe" "$MODEL_REVISION" MODEL_REVISION
  need "$recipe" "qwen38-lsa:sglang-${SGLANG_COMMIT:0:7}-lsa-${LSA_COMMIT:0:7}" "the image tag"
  need "$recipe" "\"rope_parameters\":$ROPE_PARAMETERS" ROPE_PARAMETERS
done
if [ -f qwen38-lsa.yaml ]; then
  need qwen38-lsa.yaml "/maps/$MAP_FILE" MAP_FILE
  # Comments, the description and the override line may differ. Nothing else may.
  a="$(grep -vE "^#|description:|override-args" qwen38-baseline.yaml)"
  b="$(grep -vE "^#|description:|override-args" qwen38-lsa.yaml)"
  [ "$a" = "$b" ] || { echo "check-recipes.sh: the recipes differ beyond the lsa block" >&2; fail=1; }
fi
exit "$fail"
