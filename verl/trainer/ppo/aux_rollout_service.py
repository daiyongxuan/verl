from omegaconf import DictConfig
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.single_controller.ray import RayClassWithInitArgs
import zmq
import ray
import threading
import pickle
import time

class AuxRolloutService:
  def __init__(self, config: DictConfig, service_port: int = 5555):
    self.config = config
    self.service_port = service_port
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.REP)
    self.socket.bind(f"tcp://*:{service_port}")
    
    # 初始化推理资源
    self._init_rollout_workers()
    
    # 权重同步
    self.last_weight_update_time = 0
    self.weight_update_lock = threading.Lock()
    
    print(f"辅助推理服务已启动，监听端口: {service_port}")
    
  def _init_rollout_workers(self):
    # 目前只考虑 Megatron-vllm 这一种 case
    from verl.workers.megatron_workers import ActorRolloutRefWorker
    from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
    ray_worker_group_cls = NVMegatronRayWorkerGroup
    
    role_worker_mapping = {
      Role.ActorRollout: ray.remote(ActorRolloutRefWorker)
    }
    
    aux_pool_id = "aux_rollout_pool"
    resource_pool_spec = {
      aux_pool_id: [self.config.aux_rollout.n_gpus_per_node] * self.config.aux_rollout.nnodes,
    }
    
    mapping = {
      Role.ActorRollout: aux_pool_id,
    }
    
    resource_pool_manager = ResourcePoolManager(
      resource_pool_spec=resource_pool_spec,
      mapping=mapping
    )
    
    resource_pool_manager.create_resource_pool()
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    
    rollout_cls = RayClassWithInitArgs(
      cls=role_worker_mapping[Role.ActorRollout],
      config=self.config.actor_rollout_ref,
      role="rollout", # 仅用于推理
    )
    
    self.rollout_wg = ray_worker_group_cls(
      resource_pool=resource_pool,
      ray_cls_with_init=rollout_cls
    )
    
    # 初始化模型
    # 此方法由 ActorRolloutRefWorker 定义，被本类继承
    self.rollout_wg.init_model()
    
  def update_weights(self, weight_data: bytes) -> bool:
    with self.weight_update_lock:
      try:
        weight_dict = pickle.loads(weight_data)
        success = self.rollout_wg.apply_weights_from_aux_rollout(weight_dict)
        
        if success:
          self.last_weight_update_time = time.time()
          print(f"权重更新成功,时间: {self.last_weight_update_time}")
          return True
        else:
          print("权重更新失败")
          return False
      except Exception as e:
        print(f"权重更新失败: {e}")
        return False
  
  def generate_sequences(self, batch_data: bytes) -> bytes:
    try:
      gen_batch = pickle.loads(batch_data)
      output_batch = self.rollout_wg.generate_sequences(gen_batch)
      result_data = pickle.dumps(output_batch)
      return result_data
    except Exception as e:
      print(f"推理执行失败: {e}")
      error_result = {"error": str(e)}
      return pickle.dumps(error_result)
  
  def run(self):
    """运行服务主循环"""
    print("辅助推理服务开始运行……")
    while True:
      try:
        message = self.socket.recv()
        request = pickle.loads(message)
        
        command = request.get("command")
        
        if command == "update_weights":
          success = self.update_weights(request["weight_data"])
          response = {"success", success}
        elif command == "generate":
          result_data = self.generate_sequences(request["batch_data"])
          response = {"result", result_data}
        elif command == "ping":
          response = {"status": "ok", "last_update": self.last_weight_update_time}
        else:
          response = {"error": f"未知命令 {command}"}
        # 发送响应
        self.socket.send(pickle.dumps(response))
      except Exception as e:
        print(f"处理请求时发生错误: {e}")
        error_response = {"error", str(e)}
        self.socket.send(pickle.dumps(error_response))
  def shutdown(self):
    print("正在关闭辅助推理服务")
    self.socket.close()
    self.context.term()

def main():
  import argparse
  from omegaconf import OmegaConf
  
  parser = argparse.ArgumentParser(description="辅助推理服务")
  parser.add_argument("--config", type=str, required=True, help="配置文件路径")
  parser.add_argument("--port", type=int, default=5555, help="服务端口")
  args = parser.parse_args()
  
  # 加载配置
  config = OmegaConf.load(args.config)
  
  # 初始Ray
  ray.init()
  
  service = AuxRolloutService(config, args.port)
  
  try:
    service.run()
  except KeyboardInterrupt:
    print("收到中断信号,关闭服务……")
  finally:
    service.shutdown()

if __name__ == "__main__":
  main()