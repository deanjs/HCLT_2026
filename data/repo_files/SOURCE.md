# 실파일 출처 (step A-2 선행 코드)

step A-2는 **실제 저장소 파일**을 선행 코드로 쓴다. 아래 파일들은 허용 라이선스
저장소에서 **원문 그대로** 내려받아 번들한 것이다(수정 없음). 언어별 자연 표기
관습(Python=snake, JavaScript=camel)을 자연 실험으로 이용한다.

scaleup-500에서 파일 수를 확장했다(Python 37 + JavaScript 40 = 77).
선정 기준: (1) 두 단어 이상 이름이 드러나는 파일(한 단어만 있으면 판정 불가 `other`라 제외),
(2) 파일 크기 Python 2.5~25KB / JavaScript 0.4~25KB(모델 컨텍스트 초과 방지).
파일럿 6개(fnmatch·textwrap·string·utils·buildURL·formDataToJSON)는 재현성을 위해 무조건 포함.

## Python (관습: snake_case) — 37개

출처: **CPython** — https://github.com/python/cpython (tag `v3.11.0`)
라이선스: **PSF License Agreement** (재배포 허용, 저작권·라이선스 고지 유지)

| 파일 | 원본 경로 |
|---|---|
| `python/base64.py` | `Lib/base64.py` |
| `python/bisect.py` | `Lib/bisect.py` |
| `python/cProfile.py` | `Lib/cProfile.py` |
| `python/cmd.py` | `Lib/cmd.py` |
| `python/codeop.py` | `Lib/codeop.py` |
| `python/colorsys.py` | `Lib/colorsys.py` |
| `python/copy.py` | `Lib/copy.py` |
| `python/csv.py` | `Lib/csv.py` |
| `python/filecmp.py` | `Lib/filecmp.py` |
| `python/fileinput.py` | `Lib/fileinput.py` |
| `python/fnmatch.py` | `Lib/fnmatch.py` |
| `python/getopt.py` | `Lib/getopt.py` |
| `python/getpass.py` | `Lib/getpass.py` |
| `python/gettext.py` | `Lib/gettext.py` |
| `python/graphlib.py` | `Lib/graphlib.py` |
| `python/gzip.py` | `Lib/gzip.py` |
| `python/hmac.py` | `Lib/hmac.py` |
| `python/imghdr.py` | `Lib/imghdr.py` |
| `python/mimetypes.py` | `Lib/mimetypes.py` |
| `python/modulefinder.py` | `Lib/modulefinder.py` |
| `python/netrc.py` | `Lib/netrc.py` |
| `python/opcode.py` | `Lib/opcode.py` |
| `python/pipes.py` | `Lib/pipes.py` |
| `python/poplib.py` | `Lib/poplib.py` |
| `python/pprint.py` | `Lib/pprint.py` |
| `python/profile.py` | `Lib/profile.py` |
| `python/pyclbr.py` | `Lib/pyclbr.py` |
| `python/queue.py` | `Lib/queue.py` |
| `python/selectors.py` | `Lib/selectors.py` |
| `python/shlex.py` | `Lib/shlex.py` |
| `python/sndhdr.py` | `Lib/sndhdr.py` |
| `python/string.py` | `Lib/string.py` |
| `python/stringprep.py` | `Lib/stringprep.py` |
| `python/symtable.py` | `Lib/symtable.py` |
| `python/tabnanny.py` | `Lib/tabnanny.py` |
| `python/telnetlib.py` | `Lib/telnetlib.py` |
| `python/textwrap.py` | `Lib/textwrap.py` |

## JavaScript — axios (관습: camelCase) — 14개

출처: **axios** — https://github.com/axios/axios (tag `v1.6.0`)
라이선스: **MIT** (저작권·라이선스 고지 유지 조건 재배포 허용)

| 파일 | 원본 경로 |
|---|---|
| `javascript/Axios.js` | `lib/core/Axios.js` |
| `javascript/AxiosError.js` | `lib/core/AxiosError.js` |
| `javascript/AxiosHeaders.js` | `lib/core/AxiosHeaders.js` |
| `javascript/AxiosURLSearchParams.js` | `lib/helpers/AxiosURLSearchParams.js` |
| `javascript/InterceptorManager.js` | `lib/core/InterceptorManager.js` |
| `javascript/buildFullPath.js` | `lib/core/buildFullPath.js` |
| `javascript/buildURL.js` | `lib/helpers/buildURL.js` |
| `javascript/cookies.js` | `lib/helpers/cookies.js` |
| `javascript/dispatchRequest.js` | `lib/core/dispatchRequest.js` |
| `javascript/formDataToJSON.js` | `lib/helpers/formDataToJSON.js` |
| `javascript/http.js` | `lib/adapters/http.js` |
| `javascript/isURLSameOrigin.js` | `lib/helpers/isURLSameOrigin.js` |
| `javascript/mergeConfig.js` | `lib/core/mergeConfig.js` |
| `javascript/utils.js` | `lib/utils.js` |

## JavaScript — lodash (관습: camelCase) — 26개

출처: **lodash** — https://github.com/lodash/lodash (tag `4.17.21-npm`)
라이선스: **MIT** (저작권·라이선스 고지 유지 조건 재배포 허용)

| 파일 | 원본 경로 |
|---|---|
| `javascript/cloneDeep.js` | `cloneDeep.js` |
| `javascript/cloneDeepWith.js` | `cloneDeepWith.js` |
| `javascript/defaultsDeep.js` | `defaultsDeep.js` |
| `javascript/differenceBy.js` | `differenceBy.js` |
| `javascript/dropRight.js` | `dropRight.js` |
| `javascript/findIndex.js` | `findIndex.js` |
| `javascript/findLastIndex.js` | `findLastIndex.js` |
| `javascript/flatMap.js` | `flatMap.js` |
| `javascript/flattenDeep.js` | `flattenDeep.js` |
| `javascript/fromPairs.js` | `fromPairs.js` |
| `javascript/groupBy.js` | `groupBy.js` |
| `javascript/intersectionBy.js` | `intersectionBy.js` |
| `javascript/invertBy.js` | `invertBy.js` |
| `javascript/isArrayLike.js` | `isArrayLike.js` |
| `javascript/isEmpty.js` | `isEmpty.js` |
| `javascript/isEqual.js` | `isEqual.js` |
| `javascript/isEqualWith.js` | `isEqualWith.js` |
| `javascript/isPlainObject.js` | `isPlainObject.js` |
| `javascript/isTypedArray.js` | `isTypedArray.js` |
| `javascript/keyBy.js` | `keyBy.js` |
| `javascript/mapKeys.js` | `mapKeys.js` |
| `javascript/mapValues.js` | `mapValues.js` |
| `javascript/meanBy.js` | `meanBy.js` |
| `javascript/mergeWith.js` | `mergeWith.js` |
| `javascript/omitBy.js` | `omitBy.js` |
| `javascript/orderBy.js` | `orderBy.js` |

## 사용 규약

- 파일은 **읽기 전용 선행 코드**로만 쓰며 내용을 조작하지 않는다(§4: 실코드 무조작).
- 개별 파일에 인라인 라이선스 헤더가 없더라도, 라이선스 원문은 각 상위 저장소(LICENSE)를
  따르며 위 표에 출처·태그·라이선스를 명시해 고지를 유지한다.
- 연구 재현 목적의 인용·번들이다.
