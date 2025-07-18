# Verl Rollout 代码实现

整个 ActorTraining -> Rollout 的大循环，在类 RayPPOTrainer 的 fit 方法上。之后调用其 worker Group 的 generate 方法即可。

如果要动态地增加资源的话，有两个方案，一个是增加 actor_rollout_wg 的 worker 数量，另一个方案是增加一组 aux_actor_rollout_wg，在 generate 的时候调用两次。actor_rollout_wg 的格式是 RayWorkerGroup。所以我们要是需要加一组辅助的推理资源的时候，需要额外创建一个 WorkerGroup，不一定要用 Ray来托管。

我目前更倾向于第二种方案。

在调用 generate_sequences 方法的时候，里面的参与 gen_batch 是 prompt，不过被 DataProto 包装了一下，里面加包含着一些别的信息。

新创建一组 aux_actor_rollout_wg，需要额外实现一些方法。generate 这个应该没有什么需要改动的，update_actor 这个方法的作用是对模型的权重进行更新，也就是 actor training。

Rollout 更新权重的时候，是在 generate 方法里面。with self.sharding_manager 语句之中，会更新 Rollout/vllm 的模型参数。ShardingManager 为MegatronVLLMShardingManager。在更新权重的时候，核心的操作是调用loaded_params = model.load_weights(per_tensor_param)，这个 model 是一个 vllm 实例，然后 load_weights 是 vllm 的功能。

如果需要额外增加一组 Rollout 资源的话，应该创建一个 Worker 类。选择 Megatron 策略的话应该使用 Megatron 版本的 ActorRolloutRefWorker，并且指定 role 参数为 Rollout。这个 Worker 是在 RayPPOTrainer 里面的 init_workers 里面创建的。

Worker 的实际创建是在 RayWorkerGroup 之中的 _init_with_resource_pool之中，通过一个循环创建的。

