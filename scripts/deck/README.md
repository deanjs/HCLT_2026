# 발표 덱 생성기 — `docs/발표/HCLT2026_실험정리.pptx`

실험 전체를 93장으로 정리한 덱을 **결과 파일에서 직접** 만든다.
손으로 적은 수치·손으로 그린 그림은 없다(§6).

## 다시 만들기

```bash
cd scripts/deck
python stats.py          # results/ 를 읽어 stats.json 생성 (수치의 유일한 출처)
python eqs.py            # 수식 20종 → assets/eq_*.png
python dia_concept.py; python dia_plan.py; python dia_harness.py; python dia_steps.py
bash render.sh           # build.py 실행 → pptx → pdf → qa/sheet*.png (연락지)
```

## 구성

| 파일 | 무엇 |
|---|---|
| `stats.py` | `results/` 전수 집계 → `stats.json`. 슬라이드의 모든 숫자가 여기서 온다 |
| `kit.py` | 도해·수식 이미지 공통 도구(다크, 투명 배경) |
| `dia_*.py` | 도해 생성 — 개념 8 · 계획 7 · 하네스 7 · 스텝 10 |
| `eqs.py` | 수식 20종 (matplotlib mathtext, Computer Modern) |
| `deck.py` | 디자인 시스템 — 고정 앵커, 카드, 정의목록, 통계, 표 |
| `lay.py` | 반복 배치(4연속 소형다중, 표, 코드 패널) |
| `build.py` | 93장 본문 |

## 넘침 방지

`deck.py`는 Inter/IBM Plex Mono 실측 메트릭으로 줄 수를 미리 계산한다.
본문·정의목록·표·코드·제목·눈썹이 카드를 넘치면 **빌드가 예외로 멈춘다.**
줄어들 수 있는 것(표 글자, 코드 글자)은 자동 축소하고, 그래도 안 되면 멈춘다.

## step5

step5는 **통제 전층 스윕이 끝난 뒤** 슬라이드를 붙인다.
지금 덱에는 step5 결과가 없고, 종합부에 "다음 할 일" 한 장만 있다.

## 폰트

Inter · IBM Plex Mono가 필요하다. 없으면 렌더러가 대체 폰트를 쓰므로
줄바꿈 계산과 실제 렌더가 어긋난다.
