"""
aux Rollout 客户端,用于与独立的辅助推理服务通信
"""

import zmq
import threading
import pickle

class AuxRolloutClient:
  def __init__(self, service_host: str, service_port: int=5555, timeout: int = 30):
    self.service_host = service_host
    self.service_port = service_port
    self.timeout = timeout * 1000
    
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.REQ)
    self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
    self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
    
    # connect to aux rollout service
    self.socket.connect(f"tcp://{service_host}:{service_port}")
    # lock
    self.connection_lock = threading.Lock()
    
    print(f"辅助推理客户端已连接到 {service_host}:{service_port}")
    
  def _send_request(self, request: dict) -> dict:
    with self.connection_lock:
      try:
        request_data = pickle.dumps(request)
        
        # send request
        self.socket.send(request_data)
        
        # receive response
        response_data = self.socket.recv()
        response = pickle.loads(response_data)
        
        return response
      except zmq.error.Again:
        raise TimeoutError(f"请求超时,服务器可能不可用: {self.service_host}:{self.service_port}")
      except Exception as e:
        raise RuntimeError(f"与aux Rollout 资源通信失败: {e}")
  
  def is_available(self):
    try:
      response = self.ping()
      return response.get('status') == 'ok'
    except Exception as e:
      return False
  
  def update_weights(self, weight_dict: dict) -> bool:
    try:
      weight_data = pickle.dumps(weight_dict)
      request = {
        "command": "update_weights",
        "weight_data": weight_data
      }
      
      response = self._send_request(request)
      
      if "success" in response:
        return response["success"]
      else:
        print(f"权重更新失败: {response.get('error', '未知错误')}")
        return False
    except Exception as e:
        print(f"权重更新过程中发生错误: {e}")
        return False

  def ping(self):
    request = {"command", "ping"}
    return self._send_request(request)
    
class AuxRolloutManager:
  def __init__(self, service_hosts: list[str], service_port: int = 5555):
    self.service_hosts = service_hosts
    self.service_port = service_port
    self.clients: list[AuxRolloutClient] = []
    
    for host in service_hosts:
      try:
        client = AuxRolloutClient(host, service_port)
        if client.is_available():
          self.clients.append(client)
          print(f"成功连接到辅助推理服务: {host}:{service_port}")
        else:
          print(f"无法连接到辅助推理服务: {host}:{service_port}")
      except Exception as e:
        print(f"连接辅助推理服务失败 {host}:{service_port}: {e}")
    print(f"总共连接了 {len(self.clients)} 个辅助推理服务")
  
  def update_weights(self, weight_dict: dict) -> bool:
    for client in self.clients:
      if client.update_weights(weight_dict):
        success_count += 1
    print(f"权重更新成功: {success_count}/{len(self.clients)}")
    return success_count > 0
    
  @property
  def is_available(self) -> bool:
    return len(self.clients) > 0 and any(client.is_available() for client in self.clients)
    