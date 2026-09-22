# Refactoring Plan: Modularity & Code Consolidation

이 문서는 `flck` 파일 길이 린트(hard limit 300줄)를 준수하고, 헥사고날 아키텍처 및 단일 책임 원칙(SRP)을 확립하기 위한 체계적인 모듈형 분리 및 통합 리팩터링 계획서입니다.

---

## 1. 개요 및 배경

- **린트 기준**: `soft = 250`, `hard = 300` (라인 수)
- **`flck -v` 검사 결과 현황**:
  - `ERROR` 996줄: `src/latentmas_reprop/application/intervention_use_case.py`
  - `ERROR` 825줄: `src/latentmas_reprop/application/receiver_acquisition_use_case.py`
  - `ERROR` 746줄: `src/latentmas_reprop/domain/services/prompts.py`
  - `ERROR` 553줄: `src/latentmas_reprop/infrastructure/models/model_wrapper.py`
  - `ERROR` 540줄: `tests/test_intervention.py`
  - `ERROR` 498줄: `src/latentmas_reprop/domain/services/latent_mas.py`
  - `ERROR` 395줄: `src/latentmas_reprop/domain/models.py`
  - `ERROR` 304줄: `src/latentmas_reprop/infrastructure/tracking/mlflow_tracker.py`
  - `WARN`  297줄: `src/run/cli.py` (300줄 임박)

기계적인 파일 쪼개기로 인한 **코드베이스 파편화(Fragmentation)**를 방지하고, 기존 모듈의 역할을 복원하기 위해 **3대 리팩터링 접근법**을 적용합니다:
1. **파일 분리만으로 충분 (Pure Decomposition)**: 독립된 단일 책임을 서브 모듈로 분할.
2. **기존 파일에 일부 통합 (Re-homing / Consolidation)**: 본래 있어야 할 기존 모듈로 책임을 이관/흡수.
3. **공통부 추출 및 모듈화 (Commonality Extraction)**: 파일 간 복사-붙여넣기된 중복 패턴을 공통 컴포넌트로 통합 신설.

---

## 2. 3대 영역별 상세 실행 계획

### 📁 카테고리 1: 파일 분리만으로 충분한 대상 (Pure Decomposition)

다른 모듈과의 공유나 중복 없이, 해당 파일 내부의 독립적 서브 책임을 파일 단위로 격리하는 작업입니다.

| 대상 파일 | 분리 대상 책임 | 분리 결과 및 경로 |
|---|---|---|
| **`model_wrapper.py`** | Latent Realignment 행렬 계산, Ridge 회귀, `.cache/models/realign/` 디스크 캐싱 (`_build_latent_realign_matrix` 등 약 130줄) | `src/latentmas_reprop/infrastructure/models/realignment.py` 로 추출 |
| **`domain/models.py`** | 코어 모델 vs Intervention 모델 vs Receiver Acquisition 모델의 바운디드 컨텍스트 혼재 | `src/latentmas_reprop/domain/models/` 패키지화<br>- `core.py`: `Agent`, `ProblemSample`, `BenchmarkMetrics`<br>- `intervention.py`: `InterventionMetrics`, `SampleInterventionRecord`<br>- `acquisition.py`: `ReceiverAcquisitionMetrics`, `ReceiverAcquisitionRecord` |
| **`test_intervention.py`** | `generate_cross_indices` 순열 알고리즘 단위 테스트 vs UseCase 파이프라인/트래커 통합 테스트 | - `tests/test_intervention_algo.py`<br>- `tests/test_intervention_use_case.py` |
| **`receiver_acquisition_use_case.py`** | `SecretDigitSample` 생성, 셔플링 및 숫자 교차 페어링 로직 (`generate_secret_digit_samples`, `pair_different_digit_sources` 약 60줄) | `src/latentmas_reprop/application/receiver_acquisition/sampling.py` |
| **`receiver_acquisition_use_case.py`** | 로짓 스코어링 및 단일 토큰 유효성 검증 로직 (`validate_digit_candidates`, `score_digit_logits` 약 100줄) | `src/latentmas_reprop/application/receiver_acquisition/scoring.py` |
| **`src/run/cli.py` (WARN)** | 기본 벤치마크 인자 파싱 vs 개입/실험 전용 CLI 인자 (`--intervention`, `--cross_policy` 등) | `src/run/cli_intervention.py` 로 서브 인자 등록 함수 분리 |

---

### 📥 카테고리 2: 기존 파일에 일부 통합해야 할 대상 (Re-homing / Consolidation)

새 파일을 추가하지 않고, 이미 존재하는 기존 모듈의 원래 역할에 맞춰 책임을 이관/흡수시키는 작업입니다.

| 현재 위치 | 이관할 코드 내용 | 통합 대상 (기존 파일) | 기대 효과 및 정합성 |
|---|---|---|---|
| **`latent_mas.py`** | `run_batch_vllm` 메서드 (약 180줄)<br>- Qwen 태그 검색 및 prompt embedding 텐서 결합, vLLM 호출 | 기존 `src/latentmas_reprop/infrastructure/models/model_wrapper.py` | 도메인이 저수준 텐서 조작 및 vLLM 엔진을 직접 건드리는 계층 침범 해결. `model.generate_with_latent_embeddings(...)` 인터페이스로 캡슐화되어 `latent_mas.py` 약 180줄 즉시 감소. |
| **`mlflow_tracker.py`** | `get_git_commit_hash` 함수 (18줄) | 기존 `src/latentmas_reprop/infrastructure/paths/resolver.py` | 프로젝트 루트 경로를 이미 관리하는 `PathResolver.get_git_commit_hash()`로 흡수. 새 파일 생성 없이 린트(300줄 하드 리밋) 즉시 통과. |
| **`receiver_acquisition`** | `SENDER_PROMPT_TEMPLATE`, `build_sender_messages`, `build_receiver_messages` (약 30줄) | 기존 `src/latentmas_reprop/domain/services/prompts.py` | 유스케이스 내부에 부유하는 프롬프트를 도메인 프롬프트 서비스의 단일 관리 창구로 통합. |
| **`test_intervention.py`** | `DynamicCache` 텐서 연산 및 Truncation 테스트 (약 180줄) | 신규 모듈 `domain/services/kv_cache.py` 대응 테스트 파일 `tests/test_kv_cache.py` | 이미 모듈화된 `kv_cache.py`의 전용 테스트 슈트를 정식 구축하고, `test_intervention.py` 라인 수 절감. |

---

### 🧩 카테고리 3: 공통부 추출 및 모듈화 (Commonality Extraction)

여러 파일(기존 파일 + 신규 `receiver_acquisition`) 간에 구조적으로 중복된 패턴을 공통 컴포넌트로 추출하여 중복 코드를 제거하는 작업입니다.

#### ① [공통 추출] 실험 결과 아티팩트 저장기 (`ExperimentReporter`)
- **중복 발생 파일**:
  - `application/benchmark_use_case.py`
  - `application/intervention_use_case.py` (Phase 4: 약 60줄)
  - `application/receiver_acquisition_use_case.py` (Phase 4: 약 60줄)
- **중복 내용**:
  - `run_stem` 파일명 생성 규칙 (`{task}_{model}_s{n}_{timestamp}`)
  - `sample_results.jsonl`, `summary.json`, `resolved_config.yaml` 파일 쓰기
  - MLflow `log_artifact(..., artifact_path="results")` 업로드 호출
- **추출 모듈**: `src/latentmas_reprop/application/common/reporter.py`
- **효과**: 세 유스케이스에서 파일 시스템 I/O 및 직렬화 코드 **총 150줄 이상 제거**.

#### ② [공통 추출] 다차원 지표/통계 집계 서비스 (`EvaluationService` 확장)
- **중복 발생 파일**:
  - `application/evaluation_service.py` (현재 단 7줄로 방치)
  - `application/intervention_use_case.py` (Phase 3: 통계, 전이 행렬, McNemar 등 약 180줄)
  - `application/receiver_acquisition_use_case.py` (`aggregate_receiver_records`: 셀 그룹핑, 평균 마진/확률 집계 등 약 100줄)
- **중복 내용**: 실행 레코드 리스트를 순회하며 그룹별 정확도, 전이 행렬, 통계 검정을 집계하고 Metrics 객체를 빌드하는 패턴.
- **추출 모듈**: 기존 `application/evaluation_service.py`를 확장하여 공통 통계 집계 엔진으로 격상.
- **효과**: 유스케이스들에서 통계 집계 루프 **총 280줄 이상 제거**.

#### ③ [공통 추출] 태스크별 서식 제약문 템플릿 (`TaskConstraintFormatter`)
- **중복 발생 파일**:
  - `domain/services/prompts.py` 내의 5개 함수 전체 (`sequential_latent_mas`, `hierarchical_text_mas`, `single_agent` 등)
- **중복 내용**: 수학(`\boxed{}`), 객관식(`\boxed{A}`), 파이썬 코드(` ```python `) 등의 서식 요구문구가 함수마다 거대한 if-elif로 90% 중복.
- **추출 모듈**: `src/latentmas_reprop/domain/services/prompts/task_constraints.py`
- **효과**: `prompts.py` 내부의 복사-붙여넣기 조건문이 사라지며 **약 400줄 제거**.

#### ④ [공통 추출] MLflow Tracing / LiveSpan 세션 관리자 (`TraceSession`)
- **중복 발생 파일**:
  - `application/intervention_use_case.py` (Phase 2: 약 200줄)
  - `application/receiver_acquisition_use_case.py` (Phase 2: 약 150줄)
- **중복 내용**: `start_sample_trace` ➡️ child span 생성 ➡️ `set_token_usage` ➡️ `log_expectation` ➡️ `log_feedback`으로 이어지는 반복적인 트래킹 보일러플레이트.
- **추출 모듈**: `src/latentmas_reprop/infrastructure/tracking/trace_session.py` (또는 `tracking_port.py` 고수준 컨텍스트 매니저)
- **효과**: 유스케이스 비즈니스 로직 사이에 길게 늘어져 있던 로깅 보일러플레이트 코드 **약 350줄 제거**.

---

## 3. 단계별 실행 로드맵 (Phase-by-Phase Roadmap)

### 🚀 Phase 1: 공통 컴포넌트 추출 (중복 제거 및 기반 마련)
1. `prompts/task_constraints.py` 생성 ➡️ `prompts.py`의 태스크별 중복 서식 문구 제거 (746줄 ➡️ ~300줄).
2. `application/common/reporter.py` 생성 ➡️ 유스케이스들의 아티팩트 디스크 저장 로직 단일화.
3. `application/evaluation_service.py` 확장 ➡️ 유스케이스들의 통계/전이 행렬 계산 로직 이관.
4. `infrastructure/tracking/trace_session.py` 생성 ➡️ MLflow 트레이싱 보일러플레이트 공통화.

### 🔄 Phase 2: 기존 모듈 책임 재배치 (Re-homing)
1. `latent_mas.py`의 `run_batch_vllm` ➡️ `model_wrapper.py`의 vLLM 지원 메서드로 이관.
2. `mlflow_tracker.py`의 `get_git_commit_hash` ➡️ `PathResolver`로 이관.
3. `receiver_acquisition_use_case.py`의 프롬프트 ➡️ `prompts.py`로 이관.
4. `test_intervention.py`의 캐시 연산 테스트 ➡️ `tests/test_kv_cache.py`로 분리.

### ✂️ Phase 3: 도메인 모델 및 인프라 분리 (Pure Decomposition)
1. `infrastructure/models/model_wrapper.py`에서 `realignment.py` 추출.
2. `domain/models.py`를 `domain/models/` 패키지로 3분할 (`core.py`, `intervention.py`, `acquisition.py`).
3. `receiver_acquisition_use_case.py`에서 `sampling.py`와 `scoring.py` 추출.
4. `src/run/cli.py`에서 `cli_intervention.py` 추출하여 경고(WARN) 해소.

### ✅ Phase 4: 최종 검증 및 린트 통과
1. 전체 테스트 스위트 실행:
   ```bash
   just test
   ```
2. 포맷팅 및 파일 길이 린트 검증:
   ```bash
   just format && just lint
   ```
3. `flck` 결과 47개 파일 전체 통과 (`ERROR: 0`, `WARN: 0`) 확인.
