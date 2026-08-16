#!/usr/bin/env bash
# 통합 하네스가 '그 당시 하네스'와 같은 실험 입력을 만드는지 검증한다.
#
# 왜: 스텝별 브랜치마다 하네스가 조금씩 달랐고, 통합하면서 그것들을 합쳤다.
#     "코드를 고쳤으니 결과가 달라지는 것 아니냐"에 답하려면, 같은 조건에서
#     **프롬프트·치환 대상·후보·파일명이 한 글자도 다르지 않음**을 보여야 한다.
#
# 하는 일:
#   1) 각 step 브랜치의 src/를 그대로 꺼낸다(그 당시 하네스)
#   2) 옛 하네스와 지금 하네스로 각각 실험 입력을 생성해 해시를 뜬다
#   3) 조건별로 해시를 비교한다  → 하나라도 다르면 실패
#
# 쓰는 법:  bash scripts/verify_harness_equivalence.sh
set -euo pipefail
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

declare -a PAIRS=(
  "step1/cliff:step1_cliff"
  "step2/code-observe:step2_code-observe"
  "step3/code-cause:step3_code-cause"
  "step4/instr-observe:step4_instr-observe"
  "step5/instr-cause:step5_instr-cause"
)

fail=0
for pair in "${PAIRS[@]}"; do
  branch="${pair%%:*}"; step="${pair##*:}"
  mkdir -p "$WORK/$step"
  git archive "origin/$branch" src | tar -x -C "$WORK/$step"
  python scripts/_dump_experiment_inputs.py "$WORK/$step/src" "$step" "$WORK/${step}_OLD.json" >/dev/null
  python scripts/_dump_experiment_inputs.py "src"             "$step" "$WORK/${step}_NEW.json" >/dev/null
  n=$(python - "$WORK/${step}_OLD.json" "$WORK/${step}_NEW.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
print(sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k)))
PY
)
  if [ "$n" = "0" ]; then echo "  $step: 일치"; else echo "  $step: 다름 $n개"; fail=1; fi
done
[ "$fail" = "0" ] && echo "전부 일치 — 통합 하네스로 다시 돌려도 같은 실험이 된다" || { echo "불일치 발견"; exit 1; }
