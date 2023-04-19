# MobilenetV1 代码
## Run command
```python -m torch.distributed.launch --nproc_per_node=2 train.py ```
## DDP简介
* 分布式的model在多个进程中始终保持参数一致，互相影响
* 数据分配，每个进程获得batch中的一份数据
* 只需要在一个进程上evaluation就OK
  
