# Secret-Digit Representation Trajectory & Stepwise Communication Analysis

이 문서는 비밀 숫자(Secret Digit) 실험에서 $k=1 \dots 20$의 각 잠재 스텝별로 송신자의 KV 캐시 전송 능력과 수신자의 획득 궤적을 BF16 패리티 게이트 하에서 엄밀하게 수집하고, 60개/80개 셀의 단계별 통신 산포도(Stepwise Communication Scatter)를 도출하는 고급 분석 워크플로 명세입니다.

---

## 1. 개요 및 연구 목적

* **연구 목적**:
  - 잠재 스텝 $k=1 \dots 20$이 진행됨에 따라 송신자 내부 상태에 형성된 정보 표상(Sender Probe Effect)과 수신자가 실제로 캐시를 통해 예측을 변경하는 비율(Receiver Changed Percentage) 간의 동적 궤적(Trajectory)을 모델 스케일별(0.6B, 4B, 8B, 14B)로 비교·분석합니다.
  - 전송되는 캐시가 상위 스텝의 미래 정보를 참조하지 않도록, 각 $k$스텝 시점의 프리픽스만 순수하게 유지(`original_full_seq_len = prompt_len + k`)하여 인과적 무결성을 보장합니다.

---

## 2. 수집 및 패리티 게이트 (Collection & Parity Gate)

### 1) 실행 하드웨어 요구사항
- BF16 지원 CUDA 디바이스 필수.
- Qwen3-0.6B: 8GB 로컬 GPU에서 실행 가능 (`just run-representation-trajectory --model 0.6B`).
- Qwen3-4B / 8B / 14B: 최소 32GB 이상의 VRAM 요구 (14B는 48GB 권장). 양자화나 오프로딩 없이 순수 BF16 추론을 수행합니다.

### 2) 패리티 게이트 (Parity Gate)
- 수집 시작 전, $k=1$ 및 $k=4$ 시점에서 10개 표준 샘플에 대해 완전히 새로 생성된 독립 롤아웃과 KV 텐서 및 다음 토큰 로짓이 100% 일치하는지 비트 단위로 검증합니다.
- 패리티 검증 실패 시 어떤 오버라이드로도 수집을 진행할 수 없습니다.

```bash
# 10개 샘플 패리티 및 스모크 테스트
just run-representation-trajectory --smoke

# 0.6B 모델 전체 100샘플 수집 (로컬 8GB GPU)
just run-representation-trajectory --model 0.6B

# 전체 모델 순차 수집
just run-representation-trajectory
```

---

## 3. 오프라인 송신자 프로브 분석 (Sender Probe Analysis)

수집된 각 스위프의 `sender_latent_states.pt`를 모델 추론 없이 MLflow에서 직접 분석합니다:

```bash
just analyze-representation-sender-probe \
  --source-run-id <SWEEP_RUN_ID> \
  --backend torch --permutations 5000
```

- 동일한 `latentmas/representation/trajectory` MLflow 실험에 `phase=sender_probe` 태그로 프로브 결과가 기록됩니다.
- 20개 모든 스텝에 대해 5,000회 GPU 순열 검정을 거쳐 FWER 보정 p-value를 산출합니다.

---

## 4. 단계별 모델 간 통신 분석 (`analyze-stepwise-communication`)

`just analyze-stepwise-communication`은 MLflow에 저장된 완성 런 아티팩트만을 읽어 60셀(0.6B, 4B, 8B) 또는 80셀(+14B)의 정밀 산포도와 진단 곡선을 생성합니다:

```bash
just analyze-stepwise-communication \
  --probe-run-ids PROBE_06B PROBE_4B PROBE_8B \
  --sweep-run-ids SWEEP_06B SWEEP_4B SWEEP_8B \
  --independent-run-ids OWN_06B_1 OWN_06B_4 OWN_06B_20 \
    OWN_4B_1 OWN_4B_4 OWN_4B_20 OWN_8B_1 OWN_8B_4 OWN_8B_20
```

### 산출 아티팩트:
1. `stepwise_communication_metrics.csv`: 60개 또는 80개 전체 셀의 송신자 효과(pp), 수신자 변화율(%), FWER 유의성 여부.
2. `stepwise_receiver_parity.csv`: 독립 실행과의 패리티 검증 분율 및 일치 여부.
3. `stepwise_communication_scatter.png` / `.pdf`: 스텝 순서 연결선, 유의점(채워진 원), 비유의점(빈 원)으로 표현된 2차원 통신 산포도.
4. `stepwise_communication_diagnostics.png` / `.pdf`: 모델별 비평활 스텝 곡선.
