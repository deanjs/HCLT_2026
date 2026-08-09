# 실파일 출처 (step A-2 선행 코드)

step A-2는 **실제 저장소 파일**을 선행 코드로 쓴다. 아래 파일들은 허용 라이선스
저장소에서 **원문 그대로** 내려받아 번들한 것이다(수정 없음). 언어별 자연 표기
관습(Python=snake, JavaScript=camel)을 자연 실험으로 이용한다.

## Python (관습: snake_case)

출처: **CPython** — https://github.com/python/cpython (tag `v3.11.0`)
라이선스: **PSF License Agreement** (재배포 허용, 저작권·라이선스 고지 유지)

| 파일 | 원본 경로 |
|---|---|
| `python/fnmatch.py` | `Lib/fnmatch.py` |
| `python/textwrap.py` | `Lib/textwrap.py` |
| `python/string.py` | `Lib/string.py` |

## JavaScript (관습: camelCase)

출처: **axios** — https://github.com/axios/axios (tag `v1.6.0`)
라이선스: **MIT** (저작권·라이선스 고지 유지 조건 재배포 허용)

| 파일 | 원본 경로 |
|---|---|
| `javascript/utils.js` | `lib/utils.js` |
| `javascript/buildURL.js` | `lib/helpers/buildURL.js` |
| `javascript/formDataToJSON.js` | `lib/helpers/formDataToJSON.js` |

## 사용 규약

- 파일은 **읽기 전용 선행 코드**로만 쓰며 내용을 조작하지 않는다(§4: 실코드 무조작).
- 라이선스 원문은 각 상위 저장소를 따른다. 연구 재현 목적의 인용·번들이다.
