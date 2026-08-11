# step3 행동 실험 — 폐기 (2026-08-11)

**폐기 대상:** step3(선행 개수 확장 + snake headroom) 생성(behavioral) 결과 전체. 여기 보존된 `*.json`은 `step3/camel-8to5` 그룹 80건(8/7/6/5 × 20 seed). snake 그룹도 같은 사유로 폐기(원자료 미보존).

## 폐기 사유 — 하네스 결함 (데이터 아님)

생성 프롬프트가 함수 **이름을 고정하지 않고 설명만 준다**:

```python
# src/harness/prompt.py
f"Add a function that {task.description}."
```

- 모델이 이름을 **자유 생성**한다. `TaskSpec.words=("remove","duplicates")`(2단어)는 **모델에게 전달되지 않는 내부 메타데이터**다.
- 생성 과제 #1의 description이 `"removes duplicate items from a list, preserving order"`라, 모델이 **`remove_duplicates_preserve_order`(4단어 snake)**를 지어냈다.
- 4단어 snake 이름은 camel-target 하에서 **절대 뒤집히지 않아** 준수율이 8/7/6/5 전 구간 **0.000으로 포화**, seed 20개 전부 동일 출력.

## 왜 "개수 확장"이 아니라 결함인가

| | 첫 함수 3턴 표기 |
|---|---|
| stepA c4 s0 (camel 예시 4) | **[camel, camel, camel]** ✅ |
| step3 c8 s0 (camel 예시 8) | **[snake, snake, snake]** ❌ |

camel 예시가 **더 많은데 오히려 준수 0** → 개수 효과로 설명 불가. 원인은 **stepA 이후 생성 과제 #1이 `clamp`→`removeDuplicates`로 교체**(커밋 `26ffab0`)되며 이름 자유생성 교란이 극대화된 것. 즉 step3는 **stepA와 동일 조건이 아니다**(과제가 바뀜).

## 범위 — 무엇이 무효이고 무엇이 유효인가

- **무효(폐기):** 생성 기반 준수율 측정(stepA·step3 행동 트랙)은 이름 자유생성 교란을 안고 있다.
- **유효(무관):** RQ2 기제 트랙 — 준수 선호 점수 `S = logP(준수 후보) − logP(위반 후보)`는 **고정 후보를 teacher-forcing으로 채점**하므로 이 결함과 무관. stepC·step1·step2 결론은 그대로 유효.

## 재실행 전 선행 조건

행동 트랙을 다시 돌리려면 **프롬프트를 "단어 지정 + 표기 자유"로 수정**해야 한다:

```
"Add a function named with the words 'remove' and 'duplicates' that {설명}."
```

→ 단어·길이 고정, 표기(camel/snake)만 자유변수. 또는 행동 측정을 **S 점수 기반으로 전환**한다.
