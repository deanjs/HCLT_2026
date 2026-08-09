# step 0 — 하네스 골격

RQ1·RQ2·RQ3을 별도 스크립트가 아니라 **하나의 파이프라인 + 조건 축**으로 돌리기 위한 뼈대(CLAUDE.md §3). 실제 측정·개입 로직 없이 **인터페이스만** 고정한다. 실험 로직은 각 step이 채운다.

## 구성

```
harness/
  conditions.py   조건 스키마 (5개 축) — 단일 진실 공급원
  metrics.py      측정 지표 컨테이너
  results.py      결과 저장 규약 (불변)
  model.py        모델 로딩 래퍼 (GQA group 확인)
  runner.py       단일 진입점 run(condition)
tests/            모델 없이 도는 스모크 10종
results/          산출물 (불변, §6)
```

## 조건 축 (conditions.py)

`Condition` 하나가 실험 하나를 완전히 규정한다. `run(condition)`이 받는 유일한 입력.

| 축 | 값 |
|---|---|
| **선행 코드** | 준수 개수 `n_compliant` / 구성 `clone·distinct` / 출처 `synthetic·repo` |
| **지침** | 형식 `positive·negative` / 목표 표기 `camel·snake` / 어조 `weak·strong` / 닫힌 후보 |
| **개입** | `none·key·value·key_value·attn_amplify` / 층 / 공여(donor) / 증폭 배율 |
| **모델** | HF ID / 패밀리 / dtype / 양자화 |
| 그 외 | seed, 생성 과제 순서, 자유 태그 |

- 부정형 지침은 `token_notation`이 **위반 표기**를 가리킨다(RQ3 착안점: 지침 문장이 위반 신호를 담음).
- 무관 코드 통제는 개입 축의 `donor`(예: `unrelated_camel`)로 표현한다.

## 측정 지표 (metrics.py)

준수율(생성) · 준수 선호 점수(개입) · 어텐션 가중치(축 A) · ‖v‖(축 B 크기) · ‖av‖(축 A×B) · v 코사인 유사도(축 B 방향). 층 스윕은 `per_layer`에 층별로 담는다.

## 결과 저장 (results.py)

- 경로: `results/<step>/<조건-슬러그>.json`
- **불변**: 이미 있으면 덮어쓰기 시도 시 예외(§6). 실행 전 예측(prediction)을 함께 보존.
- 슬러그 예: `qwen2-5-coder-3b-instruct__pre-c2of12-clone-syn__ins-neg-camel-w__int-value-L25__s0`

## 단일 진입점 (runner.py)

`run(condition)` 하나가 유일한 진입점. RQ별 분기는 없고 조건 축 값에 따라 라우팅만 다르다. 파이프라인 단계와 그것을 채우는 step의 대응이 `PIPELINE`에 계약으로 고정돼 있으며, 아직 안 채운 단계는 담당 step을 담아 `StageNotImplemented`를 던진다.

| 단계 | 채우는 step |
|---|---|
| `build_preceding` / `build_instruction` / `build_prompt` / `measure_generation` | step A (RQ1) |
| `measure_attention` | step B (RQ2 관측) |
| `apply_intervention` / `measure_preference` | step C·step 1 (RQ2 개입) |

## 모델 래퍼 (model.py)

torch/transformers **지연 임포트** — 스키마·저장·테스트는 무거운 의존성 없이 동작한다. `gqa_info()`가 KV group 크기를 확인해, 치환이 반드시 group 단위로 이뤄지도록 한다(§3). 실제 사용은 step A부터.

## 검증

```bash
python3 -m pytest tests/ -q      # 10 passed
```

조건 검증 · 직렬화 왕복 · 슬러그 결정성 · 결과 불변성 · 진입점 라우팅을 모델 없이 확인한다.

## 다음

step A — 위 표의 생성 단계 4개를 채워 RQ1 절벽(A1)을 이 하네스에서 직접 재현한다.
