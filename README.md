# DistributedLearning
분산 학습과 관련된 연구 노트로, PyTorch 기준으로 설명함



## DP (Data Parallel)
- 모델을 여러 GPU에 로딩하고, 각 GPU에 다른 데이터 Batch를 처리한 후, 결과를 모아서 업데이트를 진행
- 단일 노드일 경우 사용
- [PyTorch 코드 참고](https://github.com/arisel117/DistributedLearning/blob/main/main_dp.py)


## DDP (Distributed Data Parallel)
- 모델을 여러 GPU에 로딩하고, torch.distributed를 통해 모든 GPU에서 파라미터 업데이트를 동기화함
- 여러 노드일 경우 사용 사용 가능, 단일 노드인 경우에도 추천
- [PyTorch 코드 참고](https://github.com/arisel117/DistributedLearning/blob/main/main_ddp.py)


## FSDP (Fully Sharded Data Parallel)
- 모델의 모든 파라미터를 여러 GPU에 Sharding(나누어)하여 로딩하고, 모든 GPU는 자신의 파라미터와 데이터 배치를 처리하고, 필요할 때마다 파라미터를 로드하여 사용함
  - 단, 효율적으로 잘 Sharding하는 방법이 필요한데, 이는 다양한 방법이 있음
    - pytorch의 경우, 기본적으로 [size_based_auto_wrap_policy](https://pytorch.org/docs/stable/fsdp.html)를 사용함
- 여러 노드일 경우 사용 가능
- 파라미터를 한 GPU에 올릴 수 없을 정도로 모델이 큰 경우 사용
- [PyTorch 코드 참고](https://github.com/arisel117/DistributedLearning/blob/main/main_fsdp.py)



## 연합 학습 (FL: Federated Learning)
- 분산 환경에서 개별 노드(클라이언트)가 데이터를 수집 후 학습을 진행하고, 학습된 모델을 중앙으로 업로드하여 전체적으로 모델을 개선하는 학습 방법
  - 클라이언트에서 학습하기 때문에 데이터를 중앙에 업로드 하지 않아 개인 정보의 보호가 이루어질 수 있음
  - 지속적인 학습이 가능하여 모델의 성능을 빠르게 개선 할 수 있음
  - 클라이언트에 학습을 의존하므로 안정성의 문제, 품질의 문제, 클라이언트 환경의 보안 문제, 모델 송수신 대역폭의 문제 등이 발생 할 수 있음


## 군집 학습 (SL: Swarm Learning)
- 블록체인을 활용하여 중앙 노드(서버) 없이 개별 노드(클라이언트)가 학습을 진행하여 모델을 개선하는 학습 방법
- 중앙 서버가 없기 때문에 모델을 Clear하게 개선 할 수 있음
- 클라이언트가 많아질수록 더욱 다양한 데이터로 모델을 개선 할 가능성이 높아짐
- [관련 논문 링크](https://www.nature.com/articles/s41586-021-03583-3)


