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



## FL
- 추가 설명 필요

## SL
- 추가 설명 필요


