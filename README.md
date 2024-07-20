# DistributedLearning
분산 학습과 관련된 연구 노트



## DP (Data Parallel)
- 모델을 여러 GPU에 로딩하고, 각 GPU에 다른 데이터 Batch를 처리한 후, 결과를 모아서 업데이트를 진행
- 단일 노드일 경우 사용


## DDP (Distributed Data Parallel)
- 모델을 여러 GPU에 로딩하고, torch.distributed를 통해 모든 GPU에서 파라미터 업데이트를 동기화함
- 여러 노드일 경우 사용


## FSDP (Fully Sharded Data Parallel)
- 모델의 모든 파라미터를 여러 GPU에 Sharding(나누어)하여 로딩하고, 모든 GPU는 자신의 파라미터와 데이터 배치를 처리하고, 필요할 때마다 파라미터를 로드하여 사용함
- 파라미터를 한 GPU에 올릴 수 없을 정도로 모델이 크고, 여러 노드일 경우 사용




## FL


## SL
