# step3 · snake-8to5 — headroom 상단 앵커 (RQ1)

> ⚠️ **폐기 (2026-08-11).** 이 실험의 생성 결과는 하네스 프롬프트 결함(이름 자유생성)으로 폐기됨 — 사유·재실행 조건: `results/step3_dispose/DISPOSED.md`.

**한 줄:** snake headroom의 **위쪽 구간(준수 8/7/6/5)** 에서 baseline이 실제로 높은지, 위반(camel)이 소수~절반일 때 준수가 유지되는지 확인한다. `snake-4to0`과 합쳐 **절벽 곡선의 위쪽 앵커**를 만든다.

대응: **RQ1 보강** — 절벽 곡선을 그리기 위한 **headroom 상단 기준점**. RQ3 아님(RQ3은 지침 Key/Value = step4).

## 무엇을 확인하나

절벽을 "절벽"이라 부르려면 **위쪽이 높아야** 한다(높은 데서 뚝 떨어져야 절벽). `snake-4to0`이 하락 구간이라면, 이 브랜치는 그 **위쪽 앵커**다:

- 준수(snake) 8/7/6/5 → 위반(camel) 선행이 각각 **4/5/6/7개**(소수~절반).
- 지침·prior·관습이 모두 snake로 정렬돼 있으므로, 위반이 소수인 이 구간은 **준수율이 높게 유지**될 것으로 예상.
- 이 상단이 확인돼야 `snake-4to0`의 하락과 이어 **절벽 형태**가 성립한다.

## step A와 다른 것

| 축 | step A | **step3 · snake-8to5** |
|---|---|---|
| 지침 목표 표기 | camel | **snake** (Python 관습 정렬) |
| 선행 `n_compliant` (준수=snake) | 4/3/2/1/0 | **8/7/6/5** |
| 선행 `composition` | CLONE 12 | 동일 |
| 개입 | none | none |
| 반복 | seed 20 | seed 20 |

→ 4 × 20 = **80회**. `snake-4to0`과 합치면 준수 **8..0 전체 9점** 곡선이 된다.

## 재는 것

준수율(첫 함수)·자기증폭·위반 턴 분포. `snake-4to0`과 **병합**해 8..0 곡선으로 절벽 위치를 확정.

## 예상과 해석 (§4: 조건 안 맞춤, 그대로 기록)

| 결과 | 해석 |
|---|---|
| 8/7/6/5 높게 유지(≈0.85+) | headroom 상단 확인 → 절벽은 낮은 n(=`snake-4to0`)에서. 절벽 곡선 성립 |
| 8에서도 낮음 | 위반 **소수에도** 준수 억압? 예상 밖 → 그대로 기록(snake 방향에서도 위반 민감성이 큼) |
| 8→5에서 이미 완만 하락 시작 | 절벽이 아니라 **연속 민감** → `snake-4to0`과 합쳐 곡선 모양으로 판정 |

## 설정

- 모델 `Qwen/Qwen2.5-Coder-3B-Instruct` (fp16), max_new_tokens 256, greedy — step A와 동일(비교 통제)
- 결과: `results/step3/<조건-슬러그>.json` (불변, §6). 슬러그의 `pre-c{8..5}of12` + `ins-pos-snake-w`로 조건이 드러남
- 노트북: `notebooks/step3_count-extend-snake.ipynb` (셀3 `GROUP='snake-8to5'`, §5, 재개 가능)

## 다음

`camel-8to5`·`snake-4to0`과 함께 `step3/count-extend`로 머지 → `docs/step3/results.md`에 종합(8..0 곡선, 절벽/바닥 판정).
