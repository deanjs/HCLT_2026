# step A-1 결과 — 선행 distinct (1차 실행)

**실행:** Qwen2.5-Coder-3B-Instruct (fp16), 100회 = n_compliant {4,3,2,1,0} × seed 20.
선행 = **서로 다른 과제 12개(distinct)**. 지침 = 긍정·목표 camelCase·약한 어조. 개입 없음.
결과 원본: `results/stepA-1/*.json` (불변, §6). 대조: `results/stepA/`(clone).

**대응 RQ:** RQ1 대조 — "데모 오버라이드" 반론 방어.

## 예측 (실행 전)

> 선행이 서로 다른 과제여도 clone과 유사한 바닥/primacy/잠금이면 데모 오버라이드가 아님.

## 관측 결과 — clone vs distinct

| n_compliant | clone 준수율 | **distinct 준수율** |
|---|---|---|
| 4 | 0.200 | **0.000** |
| 3 | 0.200 | **0.000** |
| 2 | 0.150 | **0.000** |
| 1 | 0.050 | **0.000** |
| 0 | 0.000 | **0.000** |

- clone 전체 턴: snake 264 / **camel 36** / other 0
- distinct 전체 턴: snake 221 / **camel 0** / other 79

distinct는 **camel을 단 한 번도 생성하지 않았고**, n_compliant 효과가 **완전히 소멸**(전 구간 0).

## 측정 한계 — "other" 79개 (clamp 과제)

모델이 고른 이름: `count_vowels` 100, `merge_dictionaries` 100, **`clamp` 79**, `clamp_number` 21.

- 표기 판정은 **구조**로 한다: camel=내부 대문자, snake=밑줄, **한 단어면 판정 불가("other")**.
- 생성 과제 1(단어 `clamp`,`number`)을 모델이 **한 단어 `clamp`로** 지어(79회) "other"가 됐다. `clamp`은 자연스럽게 한 단어가 되는 동사라, 표기 신호를 담지 못한다.
- **turn 2·3(count/merge)는 두 단어라 100% snake로 깨끗.** 즉 clamp 이슈를 빼도 distinct는 **전부 snake, camel 0**.
- clone에선 선행이 `scaleValue`(두 단어 compound)라 모델이 `clampValue`/`clamp_value`로 지어 이 문제가 없었다(클론 특유의 부작용이자, distinct에서 노출된 과제 설계 결함).

## 해석 (RQ1)

1. **distinct는 clone보다 더 깊은 바닥.** clone엔 camel 36회(주로 첫 자리 camel 예시가 salient한 arrangement)였으나 distinct는 0회.
2. **clone의 잔여 준수는 "같은 과제 데모 복사"였다.** clone 선행은 동일 과제(`scaleValue`)라 그 표기를 새 함수에 복사할 수 있었다. distinct는 복사 대상이 없어 모델이 **Python-snake prior로 완전 회귀** → camel 0.
3. 방향성은 논지와 일치: 지침(camel)보다 **언어 관습·prior(snake)**가 압도.

## 데모 오버라이드 판정 — 현재로선 불가

원래 설계는 "distinct에서도 **절벽 재현**되면 반론 무력화"였다. 그러나:

- step A(camel-on-Python)는 애초에 절벽이 아니라 **바닥**이었고, distinct는 그 바닥을 **0으로 더 깊게** 만들 뿐이다.
- 모든 조건이 바닥이라 "위반이 준수를 낮추는가"(RQ1 본질)를 볼 **headroom이 없다.**

→ **clone·distinct 비교는 headroom 있는 snake 방향에서 다시 해야 결판난다.** 현재 camel 데이터의 정직한 답은 "둘 다 바닥, distinct가 더 깊음, 형태(prior)가 지침을 압도"까지다.

## 한계

- **과제 설계 결함:** 생성 과제 1(`clamp`)이 한 단어로 축약 가능 → "other". 첫 과제를 한 단어로 못 줄이는 것으로 교체 필요.
- **바닥 regime:** camel-on-Python은 준수율 바닥이라 clone/distinct의 n_compliant 효과·절벽을 관측할 여지가 없다. 데모 오버라이드 판정은 snake-target(headroom)에서 재수행해야 한다.
- turn 1의 clamp 이슈로 distinct의 turn-1 준수율은 과소·왜곡 가능. turn 2·3은 영향 없음.

## 다음

1. **snake 방향(headroom)에서 clone·distinct 둘 다 재실행** — 절벽이 나오고 distinct도 재현하는지가 진짜 demo-override 판정.
2. **생성 과제 1(clamp) 교체** — 한 단어로 축약 불가한 과제로(예: 항상 두 단어가 되는 작업). "other" 제거.
