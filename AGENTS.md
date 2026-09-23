# Agent Instructions & Guidelines

이 문서는 이 프로젝트에서 작업하는 AI 코딩 에이전트가 반드시 따라야 할 기본 작업 지침 및 프로젝트 색인입니다.

---

## ⚠️ 필수 준수 사항 (Mandatory Rules)

### 1. 작업 시작 전: `README.md` 필수 확인

- 새로운 작업이나 기능 수정 전에는 반드시 프로젝트 루트의 [README.md](README.md)를 먼저 확인합니다.
- 프로젝트 개요, 주요 문서 진입점, 현재 레포지토리의 기본 동작 방식을 파악한 뒤 작업을 진행합니다.

### 2. 작업 마무리 전: `just` 기반 린트 및 테스트 필수 수행

- 작업 완료 및 변경 사항 커밋/종료 전, 다음 명령어를 반드시 성공적으로 통과시켜야 합니다:
  ```bash
  just lint && just test
  ```
- 린트 에러(`ruff`) 또는 실패하는 테스트(`pytest`)가 남아있는 상태로 작업을 완료해서는 안 됩니다.
- 필요 시 `just format`을 실행하여 포맷팅을 맞춥니다.

### 3. 환경 오류는 즉시 정기

- 환경 관련하여 문제가 발생하면 즉시 정지 후 사용자에게 보고
- 오류가 난 환경 문제 원인과 권장 해결방안을 제시
- 사용자 명시 없이는 환경 변경 금지, 특히 \*.lock 종류 파일은 변경 금지

### 4. 모든 Python 명령은 `uv`로 실행

- Python 스크립트, 모듈, 테스트 및 일회성 명령은 반드시 `uv run python`,
  `uv run pytest`처럼 `uv run`을 통해 실행합니다.
- 시스템의 `python` 또는 `python3` 명령을 직접 실행하지 않습니다.

---

## 📚 프로젝트 문서 인덱스 및 요약 (Documentation Index)

프로젝트 세부 사항은 [docs/](docs/) 디렉터리 내 주제별 문서를 참조합니다:

| 문서 링크                                                | 주요 내용 요약                                                                                |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [**`README.md`**](README.md)                             | 프로젝트 전체 진입점, 빠른 실행 명령어(`just run`), 업스트림 및 라이선스 정보                 |
| [**`docs/architecture.md`**](docs/architecture.md)       | 헥사고날 아키텍처(Ports & Adapters), 도메인 계층, 인프라스트럭처, 실행 계층 분리 원칙         |
| [**`docs/usage.md`**](docs/usage.md)                     | `just run -c <config_name>` 실행 인터페이스, 옵션 오버라이드, `just` 레시피 목록              |
| [**`docs/configuration.md`**](docs/configuration.md)     | YAML 설정 파일 명명 규칙(`{method}_{model}_{task}.yaml`), 파라미터 스키마, 신규 프리셋 작성법 |
| [**`docs/cache_and_paths.md`**](docs/cache_and_paths.md) | `PathResolver` 중앙 경로 관리, `.cache/` 계층 분리 (재정렬 가중치, 평가 로그)                 |
| [**`docs/development.md`**](docs/development.md)         | `uv` 패키지 관리, 테스트 및 린트 절차, 신규 데이터셋/평가자 어댑터 추가 가이드                |

---

## 🛠️ 핵심 실행 명령어 요약

```bash
# 벤치마크 실행 (기본 프리셋)
just run -c lmas/reprop/lm_gsm8k

# 벤치마크 실행 (옵션 오버라이드)
just run -c lmas/reprop/lm_gsm8k --max_samples 10

# 테스트 및 검증
just test
just lint
just format

# 캐시 정리
just clean-cache
```
