# Intervention Continuation 저장 구조 리팩터링 설계

## 범위와 현재 근거

이 문서는 `feat/intervention-continuation`의 후속 구현 계약이다. 다른 유스케이스, 모델 forward 및 디코딩 수식은 변경하지 않는다. 기존 데이터와 MLflow run은 삭제하거나 수정하지 않는다. 현재 구현의 기준점은 다음과 같다.

- `src/latentmas_reprop/application/intervention_continuation/runner.py`의 `compute_common_latent`: P₁ prefill 후 매 스텝 한 위치를 KV에 추가하고, hidden은 prompt 전체와 latent 전체를 이어 저장한다. 중간 callback은 누적 KV를 전달한다.
- `intervention_continuation/checkpoint.py`: 실험 경로의 10단위 선행 체크포인트만 검색한다. `intervention_continuation_use_case.py`는 실험별 `shared` 및 조건을 저장한다.
- `infrastructure/paths/naming.py`: 현재 경로에 latent step, seed, 디코딩 값 및 개입 설정을 함께 넣고 일부 프롬프트를 잘라 충돌할 수 있다.
- `domain/services/kv_cache.py`: sequence 축은 `-2`, tuple과 `layers` 기반 캐시를 다루며 DynamicCache는 변경 가능 객체다. `infrastructure/storage/tensor_store.py`는 텐서를 먼저 원자적으로 저장하고 `result.json`을 완료 표식으로 쓴다.
- `infrastructure/models/loader.py`는 모델을 eval 모드로 로드하고 `src/run/runner.py`는 전역 seed를 설정한다. 공통 계산에는 샘플링이 없지만 `decoder.py`는 temperature가 양수이면 전역 RNG로 multinomial을 수행한다. `top_p`는 전달되나 현재 수식에 사용되지 않는다.
- `infrastructure/storage/reader.py`, `intervention_continuation/analysis.py`는 run 경로와 샘플/조건을 통해 읽는다. `artifacts.py`는 MLflow `data_dir`, `rel_dir`을 기록하고 summary, config, samples 및 샘플 trace를 기록한다.
- `tests/test_intervention_continuation.py`, `tests/test_intervention_continuation_checkpoint.py`, `tests/test_kv_cache.py`, `tests/test_tensor_store.py`가 현재 실행, 재개, 캐시 및 reader 계약을 확인한다.

## 경로와 식별

`storage/experiments/` 아래 이 유스케이스 전용 루트를 사용한다. 기존 5단계 실험 디렉터리와 겹치지 않는다.

```text
intervention_continuation/<task>/<model-label>/<state-hash>/
  state.json
  samples/<sample-key>/
    shared/ls0/{kv.pt,hidden.pt,result.json}
    shared/ls10/{kv.pt,hidden.pt,result.json}
    shared/ls15/{kv.pt,hidden.pt,result.json}
    shared/ls20/{kv.pt,hidden.pt,result.json}
    results/ls0/<result-hash>/pe/{hidden.pt,result.json}
    results/ls20/<result-hash>/{ple,plce,plpe}/{hidden.pt,result.json}
  runs/<result-hash>/{config.json,summary.json,samples.jsonl}
```

`state-hash`와 `result-hash`는 각각 아래 명세의 정규 JSON(UTF-8, 키 정렬, 공백 없는 구분자)의 전체 SHA-256 소문자 hex다. 저장된 JSON의 원문을 항상 비교하고 불일치하면 중단한다. 축약된 사람이 읽는 이름으로 동일성을 판단하지 않는다. 모델 label은 탐색용이며 전체 모델 식별자는 해시에 포함한다. sample-key는 기존 6자리 인덱스다. 각 샘플의 `ls0/result.json`에 split, 인덱스, question, 실제 P₁ 텍스트와 token IDs를 기록하고 로드 전 현재 입력과 비교한다. 인덱스에 다른 샘플이 들어오면 덮어쓰지 않고 오류로 중단한다. gold는 계산 상태가 아니며 결과 식별 및 결과 메타데이터에 포함한다.

`state.json`의 계산 식별은 전체 모델 이름과 실제 로드된 모델/토크나이저의 불변 revision 또는 로컬 파일 내용 digest, 실제 dtype, backend, 실제 realignment 활성 여부와 적용 행렬의 내용 digest, P₁ 생성 설정(variant/custom 및 그에 영향을 주는 설정), chat template/tokenizer 식별, task와 split을 포함한다. 모델이나 tokenizer revision을 확정할 수 없으면 해당 파일 내용 digest를 계산하고, realignment 행렬을 확인할 수 없으면 재사용을 거부한다. 실제 P₁ 토큰은 샘플 단위 검증으로 확정한다. `think`, prompt 스타일 등은 실제 P₁ 빌더/템플릿에 영향을 주는 경우 반영하고, 영향을 주지 않는 P₂ 전용 설정은 제외한다. 실행 장치와 라이브러리/커널 버전은 provenance로 기록하며, 서로 다른 실행 환경의 추가 연장은 수치적 동일성을 보장할 수 없으므로 환경이 다르면 기존 상태 읽기만 허용하고 연장은 명시적으로 중단한다. 환경 판정에는 torch/transformers 버전, 장치 종류 및 attention 구현을 포함한다.

`latent_steps`, 체크포인트 간격, `max_samples`, 진행 표시, 디코딩 설정, P₂ 및 개입 조건은 state에서 제외한다. seed도 제외한다: 위 eval forward에는 RNG 사용이 없기 때문이다. 다만 모델이 학습 모드이거나 확률적 forward를 쓰는 경우 재사용/연장을 거부한다. `result-hash`는 state 참조, 목표 step, 조건별 실제 개입 텍스트 또는 token IDs와 P₂ 생성 설정, `max_new_tokens`, temperature, top_p, seed, task/gold 및 디코더 구현 식별을 포함한다. 조건별 hash를 계산하여 다른 조건의 변경이 기존 결과를 무효화하지 않게 한다. 위 트리의 `runs/<result-hash>`는 한 조건의 실행 요약 경로이며 복수 조건 실행은 조건별 요약을 남긴다. 동일 설정의 재실행은 완료 결과를 재사용한다. 다른 결과를 강제로 생성할 때는 기존 경로를 덮어쓰지 않고 새 실행용 별도 디렉터리를 명시적으로 지정한다.

seed는 결과 식별에 포함하되 현재 전역 RNG는 샘플 순서, 조건 순서, 건너뛴 결과에 영향을 받는다. 따라서 결과에 실제 실행 순서와 RNG 상태 provenance를 기록하고, 후속 구현에서는 각 샘플/조건 디코딩 시작 RNG 상태를 seed와 샘플/조건에서 안정적으로 설정한다. 이는 샘플링 수식은 바꾸지 않지만 기존 실행과 난수열이 달라질 수 있으므로 기존 결과는 재계산하지 않는다. top_p가 현재 무효라는 사실도 메타데이터에 기록하되 요청 값은 충돌 방지를 위해 식별에 포함한다.

## KV 및 체크포인트 계약

각 `lsN`의 KV는 직전 저장 경계 `(a,N]` 구간만 담는다. `ls0`은 P₁ prefill 전체이며 prompt 길이를 `p`라 하면 `lsN`은 누적 KV의 `[..., p+a:p+N, :]`이다. `hidden.pt`는 기존대로 P₁ 전체와 1..N 전체다. `result.json`은 시작·끝 step, prompt 길이, 레이어 수, 각 K/V shape/dtype, cache 타입, 필수 파일 및 상태를 기록한다. 생성 시 누적 KV 길이 `p+N`, hidden 길이 `p+N`, 각 레이어의 구간 길이를 검증한다.

10단위와 요청한 정확한 목표 N에서 저장한다. 예를 들어 15 후 20을 계산하면 `ls15`는 11..15, `ls20`은 16..20이다. 20이 먼저 있다면 15 요청은 기존 11..20을 앞 5 위치로 슬라이스해 `ls15`를 만들고, 기존 `ls20`은 수정하지 않는다. 따라서 조립은 `ls0`부터 시작하여 현재 끝점과 정확히 맞는 구간을 선택하거나, 더 큰 구간의 필요한 prefix를 읽는다. 겹치는 구간은 중복 연결하지 않는다. 저장된 hidden이 없는 중간 목표는 상위 체크포인트의 전체 hidden을 `p+N`까지 잘라 만든다. 구간은 불변이며 이후 구간이 이미 존재하더라도 변경하지 않는다. 불완전 파일은 완료 표식이 없으면 무시하고 해당 디렉터리만 안전하게 재작성한다. 유효한 완료 구간의 덮어쓰기는 금지한다. 누락/불연속 또는 검증 실패 시 조립 실패를 명시하고 마지막 유효 상태에서만 추론을 재개한다.

`kv_cache.py`에 순수 연산 `slice_past_kv(cache, start, end)`와 `concat_past_kv(segments)`를 추가한다. 반환은 독립 CPU 텐서이며 입력을 변경하지 않는다. 모든 레이어에서 K/V의 비시퀀스 shape, dtype, 레이어 수와 순서를 검증하고 `-2` 축으로 연결한다. 지원 범위는 현재 테스트되는 표준 `(key,value)` 레이어의 legacy tuple 및 Transformers `DynamicCache`의 일반 dense layers다. 원래 타입으로 복원하고 sequence length 등 캐시 내부 상태가 일관되게 구성되는지 검증한다. sliding/static/quantized/offloaded 또는 추가 상태가 필요한 캐시는 명시적으로 오류를 낸다. 임의의 `layers` 객체를 지원한다고 가정하지 않는다. 조립 결과가 원본 누적 KV와 레이어별 `torch.equal`이어야 한다.

`FileStorage`에 이 유스케이스의 경로를 받는 직접 메서드 `save_shared_segment`, `load_shared_segment`, `assemble_shared_state(state, sample, step)` 및 결과 저장/로드를 추가한다. `assemble_shared_state`는 CPU KV, 전체 hidden, 메타데이터를 반환하며 GPU 접근을 하지 않는다. 완료 표식은 기존 `save_condition_result`의 텐서 선행/JSON 마지막 규칙을 재사용한다. `ResearchDataReader`에는 run 또는 state 경로와 sample, step을 받는 `load_intervention_state` 및 조건/result hash를 받는 결과 로드를 추가하고 같은 조립 함수를 사용한다. `SingleInterventionAnalyzer`의 호출을 이 명시적 인터페이스로 변경한다. 기존 다른 유스케이스의 reader 및 FileStorage 동작은 유지한다.

CPU 선행 로드는 `ThreadPoolExecutor(max_workers=1)`로 다음 샘플의 `assemble_shared_state`만 예약하고 현재 샘플의 GPU 추론과 겹친다. future의 결과를 소비하는 시점에 오류를 전달하며, CPU 객체의 소유권은 소비자에게 넘긴다. GPU 전송과 모델 호출은 주 스레드에서만 수행한다. 공유 mutable DynamicCache를 재사용하지 않고 분기마다 clone한다. 재개 계산이 필요한 샘플은 준비된 마지막 유효 상태를 반환하고 GPU에서 이어 계산한다. 종료 시 future를 취소/정리한다. PE는 `ls0`의 마지막 prompt hidden과 KV로 기존 PLE 디코더를 호출해 별도 `pe` 결과를 기록한다. PLE/PLCE/PLPE는 정확한 요청 `lsN`에서 시작한다.

## MLflow 및 이전

새 run의 `data_dir`/`rel_dir`은 `runs/<result-hash>`를 가리키고 config에 state 상대경로, step 및 조건별 결과 경로를 명시한다. reader는 이 참조를 통해 공통 상태를 조립한다. 각 새 run은 기존과 같이 요약과 trace를 기록하되 기존 run의 tag, artifact, trace 및 결과 파일을 수정하지 않는다. 기존 run ID는 이전 manifest의 출처로 보존한다. MLflow가 비활성화되어도 로컬 결과를 읽을 수 있다.

이전은 별도 명시적 오프라인 명령으로 실행한다. 먼저 기존 `config.json`, 각 shared/조건 `result.json`, KV와 hidden, summary, samples, MLflow run 연결을 읽기 전용으로 목록화한다. 기존 경로명만으로 P₁이나 모델 revision을 추측하지 않는다. 실제 prompt token IDs, 설정, 모델/토크나이저 및 realignment의 증거가 충분하지 않거나 서로 충돌하면 해당 항목을 격리 보고하고 자동 병합하지 않는다. 누적 체크포인트 간 겹치는 KV와 hidden prefix를 레이어별 정확 비교한다. 동일 계산 식별인데 값이 다르면 어느 쪽도 선택하지 않고 충돌 보고한다. 독립 실행의 미세한 차이도 자동 병합하지 않는다.

각 검증된 누적 KV에서 `ls0`의 `[:p]`와 각 경계의 새 구간을 추출한다. 선행 체크포인트가 없더라도 더 긴 누적 KV를 10단위로 분할할 수 있고, 전체 hidden의 prefix로 각 체크포인트 hidden을 만든다. 부분 최종 step도 보존한다. 기존 조건의 result JSON과 hidden은 해당 step/조건/result hash로 복사하며 원래 trace_id, 모든 필드 및 파일을 보존한다. summary/config/samples 및 그 밖의 파일은 원본별로 새 run 디렉터리에 바이트 그대로 보존하고 출처 경로와 run ID를 manifest에 기록한다. 식별 정보가 불충분한 결과나 동일 hash에 서로 다른 결과가 있으면 자동 병합하지 않고 원본별 별도 보존 영역에 복사하여 보고한다. 현재 경로의 잘린 프롬프트 이름으로 동일성을 추정하지 않는다.

임시 디렉터리에 작성 후 원본과 조립 KV 및 hidden을 `torch.equal`로 비교하고 JSON/기타 파일의 바이트 또는 내용과 파일 목록, 출처를 검증한다. 성공한 샘플만 완료 표식을 마지막에 기록하고 원자적으로 최종 위치에 게시한다. 대상이 이미 있으면 동일 내용만 재사용하고 차이는 충돌로 중단한다. 실패 및 재실행은 원본을 변경하지 않으며 임시 파일만 정리한다. 보고서는 이전 성공, 보류, 충돌을 나열한다. 이전에는 모델 로드나 추론이 필요 없다.

## 구현 단계와 수용 기준

1. **캐시 연산** — `domain/services/kv_cache.py`, `tests/test_kv_cache.py`: tuple/DynamicCache의 구간 추출과 조립이 입력을 변경하지 않고 다양한 prompt 길이 및 부분 구간에서 원본과 정확히 같다. 미지원 캐시는 실패한다.
2. **경로와 저장** — `infrastructure/storage/tensor_store.py`, `infrastructure/paths/`의 이 유스케이스 경로 코드, `tests/test_tensor_store.py` 및 전용 테스트: 식별 충돌 방지, 완료 표식, 중단 후 재개, 15→20 및 20→15, 중복/누락 구간과 샘플 불일치를 검증한다. 다른 유스케이스 경로는 변하지 않는다.
3. **실행과 병렬 로드** — `application/intervention_continuation/{runner,checkpoint,decoder}.py`, `application/intervention_continuation_use_case.py`, 관련 테스트: ls0 PE와 각 lsN 분기, 재사용/연장, 조건별 충돌 방지, RNG 재현성 및 CPU 선행 로드 예외 전달을 검증한다. 동일 입력에서 조립 KV와 기존 누적 KV 및 결정적 디코딩 결과가 일치한다.
4. **분석과 추적** — `infrastructure/storage/reader.py`, `application/intervention_continuation/{analysis,artifacts,checkpoint}.py`, `src/run/runner.py`, reader/유스케이스 테스트: 새 run ID와 로컬 경로 모두에서 정확한 상태와 결과를 읽고 MLflow 참조를 확인한다. 기존 MLflow run은 불변이다.
5. **이전 도구와 문서** — `scripts/`의 전용 이전 명령, `tests/migrations/`의 전용 테스트, `docs/use_cases/intervention_continuation.md`, `docs/storage_policy.md`: 실제 형식의 여러 누적 체크포인트, 부분 step, 누락 파일, 충돌, 재실행을 검증한다. 원본과 기존 run을 보존하며 모델 없이 복원한다. 구현 완료 전 `just lint && just test`를 통과한다.

## 사용자 결정이 필요한 미해결 사항

없음. 불충분한 과거 provenance는 추정하지 않고 이전 보고서에서 보류한다. 보류 데이터의 출처를 확인할 자료가 이후 제공되면 그 항목만 다시 이전한다.

## 이번 세션 구현 및 다음 세션 인터페이스

계산 상태와 조건별 결과의 식별 및 경로를 분리했다. 모델·토크나이저 식별,
실제 dtype, realignment 행렬과 정규화 값, P₁ 및 task/split은 상태 식별에
포함한다. 디코딩 설정과 seed, gold, 조건별 실제 개입은 결과 식별에 포함한다.
샘플을 재사용하기 전에 ls0의 질문, split, P₁ 텍스트 및 토큰을 확인한다.
실행 환경이 다르면 상태를 읽을 수 있지만 연장할 수 없다. 기존 누적 데이터와
기존 MLflow run은 변경하지 않는다.

- `slice_past_kv(cache, start, end)`, `concat_past_kv(segments)`: 지원되는
  tuple 및 dense `DynamicCache`를 CPU에서 독립적으로 분할·조립한다.
- `FileStorage.save_shared_segment(state, sample, start, end, kv, hidden, metadata)`:
  KV 구간과 전체 hidden을 저장하며 완료 표식을 마지막에 기록한다.
- `FileStorage.load_shared_segment(state, sample, step)` 및
  `assemble_shared_state(state, sample, step)`: 후자는 CPU 누적 KV, 전체 hidden,
  메타데이터를 반환한다. 더 큰 구간의 prefix도 읽을 수 있고 겹친 값의 일치를
  검증한다.
- `FileStorage.save_intervention_result` / `load_intervention_result`는 조건별
  결과와 hidden을 다룬다. `ResearchDataReader.load_intervention_state`는 state
  상대경로 또는 run 상대경로와 sample, step을 받아 같은 조립 함수를 사용한다.
  `load_intervention_result`는 run 또는 state 경로, sample, step, result hash,
  condition을 받는다. run 경로에서는 config의 `state_path`와 `step`을 읽는다.

`ls0`와 10단위 경계 및 정확한 목표를 저장하고, 이후 연장은 마지막 유효 상태에서
시작한다. 디코더에는 기존과 같은 누적 KV와 전체 hidden을 전달한다. 결과는 현재
전역 RNG 상태 및 실행 순서를 기록한다. 샘플/조건별 RNG 고정, PE 실행, CPU/GPU
선행 로드 파이프라인 및 기존 데이터 이전은 이번 세션에 포함하지 않았다.

검증 결과: `CUDA_VISIBLE_DEVICES=3 just lint && CUDA_VISIBLE_DEVICES=3 just test`
통과 (232개 테스트). tuple/DynamicCache 분할·조립, ls0/ls10/ls15/ls20,
15→20 및 겹치는 구간, 부분 저장, 샘플 불일치, 상태·결과 식별 분리와
기존 유스케이스 테스트를 확인했다. 실제 모델 추론이나 데이터 이전은 실행하지 않았다.

후속 보완: 불완전한 ls0은 확인 가능한 샘플 식별이 일치할 때 prefill을 다시
저장하고 기존 유효 구간을 재사용한다. 공유 계산은 Python, NumPy 전역 RNG와
PyTorch CPU 및 현재 장치의 RNG 상태 변화를 검사하고, 변화가 감지되면 해당
상태를 무효화해 재사용을 막는다. 별도 사용자 정의 generator나 외부 난수원은
이 검사로 감지할 수 없다. RNG 검사 정책은 상태 식별에 포함하므로 이전 상태는
보존되지만 새 정책으로 자동 재사용하지 않는다. 20→15 유스케이스 재실행과
부분 ls0 복구를 모의 모델로 검증한다.
후속 보완 후 `CUDA_VISIBLE_DEVICES=3 just lint && CUDA_VISIBLE_DEVICES=3 just test`가
통과했으며 전체 234개 테스트가 성공했다.

## 이번 구현: 구간 재개 및 PE

PE는 `ls0`에서 직접 디코딩하며 다른 조건은 요청한 latent step에서 분기한다.
조건별 결과와 run은 각자의 step 및 결과 식별로 격리한다. 디코딩 RNG는 결과 식별과
샘플 인덱스로 고정해 재개와 실행 순서에 독립적이다. 완료 결과는 재계산하지 않는다.
CPU/GPU 선행 로드와 이전 도구는 이번 구현 범위에 포함하지 않았다.
모의 모델의 ls0 재개, 20→40 확장 및 확률적 연속 실행 동등성과 전체 235개 테스트를 확인했다.
