#!/usr/bin/env python3
"""
固定Channel數量的多模型並行基準測試工具
用戶設定的Channel數不會改變，用可載入的模型數量來處理所有Channel
"""
import argparse
import os
import sys
import threading
import time
import psutil
import json
from time import perf_counter
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import numpy as np
import cv2
import torch
from collections import deque
import queue
from torch.profiler import profile, record_function, ProfilerActivity
try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ 無法導入 ultralytics: {e}")
    print("請安裝 ultralytics: pip install ultralytics")
    sys.exit(1)


class FixedChannelMetric:
    """固定Channel性能指標收集器"""
    
    def __init__(self, channel_id: int = 0) -> None:
        self.channel_id = channel_id
        self.lock = threading.Lock()
        
        # 基本指標
        self.num_frames: int = 0
        self.total_proc_s: float = 0.0
        self.start_time = time.time()
        
        # 性能歷史
        self.processing_times: List[float] = []
        self.fps_history: List[float] = []
        
        # 檢測指標
        self.detection_counts: List[int] = []
        
        # 模型分配信息
        self.assigned_model_id: int = -1
        self.model_shared: bool = False
        self.profiling_data: Dict[str, Any] = {}

    def update(self, proc_s: float, detections: int = 0) -> None:
        """更新性能指標"""
        with self.lock:
            self.num_frames += 1
            self.total_proc_s += float(proc_s)
            self.processing_times.append(proc_s)
            
            # 計算當前FPS
            current_fps = 1.0 / proc_s if proc_s > 0 else 0.0
            self.fps_history.append(current_fps)
            
            # 檢測指標
            self.detection_counts.append(detections)

    def get_fps(self) -> float:
        """計算實際FPS"""
        with self.lock:
            if self.num_frames <= 0:
                return 0.0
            elapsed = time.time() - self.start_time
            if elapsed <= 0:
                return 0.0
            return self.num_frames / elapsed

    def get_latency_ms(self) -> float:
        """計算平均延遲（毫秒）"""
        with self.lock:
            if self.num_frames <= 0:
                return 0.0
            return (self.total_proc_s / self.num_frames) * 1000.0

    def get_throughput(self) -> float:
        """計算總吞吐量（每秒處理的幀數）"""
        with self.lock:
            elapsed = time.time() - self.start_time
            if elapsed <= 0:
                return 0.0
            return self.num_frames / elapsed

    def get_avg_detections(self) -> float:
        """計算平均檢測數量"""
        with self.lock:
            if not self.detection_counts:
                return 0.0
            return sum(self.detection_counts) / len(self.detection_counts)

class ResourceMonitor(threading.Thread):
    """資源監控器，用於在測試期間收集系統資源使用情況"""
    def __init__(self, sample_interval: float = 1.0):
        super().__init__()
        self.daemon = True
        self._stop_event = threading.Event()
        self.sample_interval = sample_interval
        
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.gpu_usage: List[float] = []
        
        self._pynvml_initialized = False
        if pynvml:
            try:
                pynvml.nvmlInit()
                self._pynvml_initialized = True
            except pynvml.NVMLError:
                print("⚠️ 無法初始化 pynvml，GPU 使用率將不會被監控。")

    def run(self) -> None:
        """在背景執行緒中定期收集資源數據"""
        while not self._stop_event.is_set():
            # 收集 CPU 和記憶體使用率
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # 收集 GPU 使用率
            gpu_percent = 0.0
            if self._pynvml_initialized:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = util.gpu
                except pynvml.NVMLError:
                    gpu_percent = 0.0 # 如果出錯，則記錄為0
            self.gpu_usage.append(gpu_percent)
            
            time.sleep(self.sample_interval)

    def stop(self) -> None:
        """停止資源監控"""
        self._stop_event.set()
        if self._pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """計算並返回資源使用的統計數據"""
        def _calculate(data: List[float]) -> Dict[str, float]:
            if not data:
                return {"average": 0.0, "min": 0.0, "max": 0.0}
            return {
                "average": float(np.mean(data)) if data else 0.0,
                "min": float(np.min(data)) if data else 0.0,
                "max": float(np.max(data)) if data else 0.0
            }

        return {
            "cpu": _calculate(self.cpu_usage),
            "memory": _calculate(self.memory_usage),
            "gpu": _calculate(self.gpu_usage)
        }


class FixedChannelBenchmark:
    """固定Channel數量的多模型並行基準測試主類"""
    
    def __init__(self, 
                 model_name: str = 'yolov8n.pt',
                 device: str = 'auto',
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.5):
        """初始化固定Channel基準測試器"""
        self.model_name = model_name
        self.device = self._parse_device(device)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 硬體規格檢測
        self.hardware_specs = self._detect_hardware_specs()
        
        print(f"🚀 固定Channel多模型並行基準測試器初始化完成")
        print(f"   • 模型: {model_name}")
        print(f"   • 設備: {self.device}")
        print(f"   • 圖片尺寸: {img_size}x{img_size}")
        print(f"   • 置信度閾值: {conf_threshold}")
        print(f"   • IoU 閾值: {iou_threshold}")
        print(f"   • 硬體規格: {self.hardware_specs}")

    def _parse_device(self, device: str) -> str:
        """解析設備配置"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device

    def _detect_hardware_specs(self) -> Dict[str, Any]:
        """檢測硬體規格"""
        specs = {
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': 0,
            'gpu_memory_gb': 0,
            'gpu_name': 'Unknown',
            'gpus': []
        }
        
        if torch.cuda.is_available():
            specs['gpu_count'] = torch.cuda.device_count()
            if specs['gpu_count'] > 0:
                # 檢測所有GPU
                for i in range(specs['gpu_count']):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        'id': i,
                        'name': gpu_props.name,
                        'memory_gb': gpu_props.total_memory / (1024**3),
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                    }
                    specs['gpus'].append(gpu_info)
                
                # 保持向後兼容性
                specs['gpu_memory_gb'] = specs['gpus'][0]['memory_gb']
                specs['gpu_name'] = specs['gpus'][0]['name']
        
        return specs


    def _test_model_loading(self, num_models: int) -> Tuple[bool, List[float], List[float]]:
        """測試載入指定數量的模型"""
        print(f"🧪 測試載入 {num_models} 個模型...")
        
        models = []
        load_times = []
        memory_usage = []
        
        try:
            for i in range(num_models):
                print(f"   🔄 載入模型 {i+1}/{num_models}...")
                
                start_time = perf_counter()
                model = YOLO(self.model_name)
                
                if self.device != 'cpu':
                    model.to(self.device)
                
                load_time = perf_counter() - start_time
                load_times.append(load_time)
                
                # 獲取記憶體使用量
                if torch.cuda.is_available():
                    mem_usage = torch.cuda.memory_allocated() / (1024**3)
                    memory_usage.append(mem_usage)
                else:
                    memory_usage.append(0.0)
                
                models.append(model)
                print(f"   ✅ 模型 {i+1} 載入完成，耗時: {load_time:.3f}秒")
            
            # 清理模型
            for model in models:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True, load_times, memory_usage
            
        except Exception as e:
            print(f"   ❌ 載入失敗: {e}")
            # 清理已載入的模型
            for model in models:
                try:
                    del model
                except:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return False, load_times, memory_usage

    def _find_max_loadable_models(self, requested_channels: int) -> Tuple[int, Dict[str, Any]]:
        """尋找最大可載入的模型數量（上限為Channel數量）"""
        print(f"\n🔍 尋找最大可載入模型數量...")
        print(f"   • 請求Channel數: {requested_channels}")
        
        # 從請求的Channel數開始往下測試
        max_models = requested_channels
        print(f"   • 起始測試模型數: {max_models}")
        
        if max_models <= 0:
            print("   ❌ 硬體規格不足以載入任何模型")
            return 0, {}
        
        # 從理論最大值開始，逐步遞減測試
        # 從理論最大值開始，逐步遞減測試
        for count in range(max_models, 0, -1):
            print(f"\n   🧪 測試 {count} 個模型...")
            
            success, load_times, memory_usage = self._test_model_loading(count)
            
            if success:
                print(f"   ✅ 成功載入 {count} 個模型")
                print(f"   • 平均載入時間: {np.mean(load_times):.3f}秒")
                print(f"   • 平均記憶體使用: {np.mean(memory_usage):.2f} GB")
                
                return count, {
                    'load_times': load_times,
                    'memory_usage': memory_usage,
                    'avg_load_time': float(np.mean(load_times)) if load_times else 0.0,
                    'avg_memory_usage': float(np.mean(memory_usage)) if memory_usage else 0.0,
                    'total_memory_usage': float(np.sum(memory_usage)) if memory_usage else 0.0
                }
            else:
                print(f"   ❌ 無法載入 {count} 個模型")
        
        print("   ❌ 無法載入任何模型")
        return 0, {}

    def _create_model_instance(self, model_id: int) -> Tuple[YOLO, float, float]:
        """創建模型實例"""
        print(f"🔄 載入模型實例 {model_id}...")
        
        start_time = perf_counter()
        
        try:
            # 創建新的模型實例
            yolo_model = YOLO(self.model_name)
            
            if self.device != 'cpu':
                yolo_model.to(self.device)
            
            load_time = perf_counter() - start_time
            
            # 獲取模型記憶體使用量
            memory_usage = 0.0
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            print(f"✅ 模型實例 {model_id} 載入完成，耗時: {load_time:.3f}秒")
            
            return yolo_model, load_time, memory_usage
            
        except Exception as e:
            print(f"❌ 模型實例 {model_id} 載入失敗: {e}")
            raise

    def predict_single_frame(self, model: YOLO, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """
        [宏觀測試用] 對單一幀進行預測，只返回檢測結果和總牆上時間（秒）。
        Profiler 已被移除，以確保執行緒安全。
        
        返回:
            detections (List[Dict]): 檢測結果
            processing_time_s (float): 總牆上時間 (秒)
        """
        t_wall_start = perf_counter()

        try:
            # 1. 執行推論 (不使用 profiler)
            with torch.inference_mode():
                results = model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    verbose=False,
                    save=False
                )

            # 2. CPU 後處理
            detections = []
            if results:
                for r in results:
                    if r.boxes is not None:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        confidences = r.boxes.conf.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy().astype(int)
                        
                        for i in range(len(boxes)):
                            detection = {
                                'class_id': int(classes[i]),
                                'confidence': float(confidences[i]),
                                'bbox': boxes[i].tolist(),
                                'class_name': r.names[int(classes[i])]
                            }
                            detections.append(detection)

            t_wall_end = perf_counter()
            processing_time_s = t_wall_end - t_wall_start
            
            return detections, processing_time_s
            
        except Exception as e:
            print(f"⚠️ 預測錯誤: {e}")
            return [], 0.0

    def _profile_model_once(self, model: YOLO) -> Dict[str, float]:
        """
        [微觀剖析用] 在主執行緒中對單一模型實例進行詳細剖析。
        這會預熱並運行多次推論，以獲取穩定的 GPU 運算/I/O 理論值。
        
        返回:
            Dict[str, float]: 包含 'gpu_compute_avg_ms', 'gpu_io_avg_ms', 'cpu_post_proc_avg_ms' 的字典
        """
        print(f"   🔬 [微觀剖析] 開始對 {self.model_name} 進行單模型理論值分析...")
        
        use_cuda = torch.cuda.is_available() and self.device != 'cpu'
        if not use_cuda:
            print("   ⚠️ [微觀剖析] 未使用 CUDA，跳過詳細剖析。")
            return {}

        # 創建一個符合 img_size 的假 (dummy) 圖像
        dummy_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        warmup_runs = 20
        profile_runs = 50
        
        results_compute: List[float] = []
        results_io: List[float] = []
        results_post_proc: List[float] = []

        try:
            # 1. 預熱 (Warm-up)
            print(f"   🔬 [微觀剖析] 執行 {warmup_runs} 次預熱...")
            with torch.inference_mode():
                for _ in range(warmup_runs):
                    _ = model.predict(source=dummy_frame, verbose=False)
            
            # 2. 剖析 (Profiling)
            print(f"   🔬 [微觀剖析] 執行 {profile_runs} 次剖析...")
            for _ in range(profile_runs):
                gpu_compute_s = 0.0
                gpu_io_s = 0.0
                
                with torch.inference_mode():
                    with profile(
                        activities=[ProfilerActivity.CUDA], # 我們只關心 CUDA 事件
                        record_shapes=False,
                        with_stack=False
                    ) as prof:
                        results = model.predict(
                            source=dummy_frame,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            imgsz=self.img_size,
                            verbose=False,
                            save=False
                        )
                
                # 提取 Profiler 數據
                for event in prof.events():
                    if "memcpy" in event.name.lower():
                        gpu_io_s += event.cuda_time_total / 1_000_000.0  # us -> s
                    elif "kernel" in event.name.lower():
                        gpu_compute_s += event.cuda_time_total / 1_000_000.0 # us -> s
                
                # 測量 CPU 後處理
                t_post_start = perf_counter()
                if results:
                    for r in results:
                        _ = r.boxes.xyxy.cpu().numpy() # 模擬後處理
                cpu_post_proc_s = perf_counter() - t_post_start
                
                results_compute.append(gpu_compute_s)
                results_io.append(gpu_io_s)
                results_post_proc.append(cpu_post_proc_s)
            
            # 3. 計算平均值並轉換為毫秒 (ms)
            avg_compute_ms = (sum(results_compute) / len(results_compute)) * 1000
            avg_io_ms = (sum(results_io) / len(results_io)) * 1000
            avg_post_proc_ms = (sum(results_post_proc) / len(results_post_proc)) * 1000
            
            result_dict = {
                "micro_gpu_compute_avg_ms": avg_compute_ms,
                "micro_gpu_io_avg_ms": avg_io_ms,
                "micro_cpu_post_proc_avg_ms": avg_post_proc_ms,
                "micro_total_avg_ms": avg_compute_ms + avg_io_ms + avg_post_proc_ms
            }
            print(f"   ✅ [微觀剖析] 完成: {result_dict}")
            return result_dict
            
        except Exception as e:
            print(f"   ❌ [微觀剖析] 失敗: {e}")
            return {}

    def benchmark_video_fixed_channels(self, 
                                      video_path: str, 
                                      duration_seconds: int = 60,
                                      requested_channels: int = 1,
                                      fixed_models: Optional[int] = None,
                                      output_file: Optional[str] = None) -> Dict[str, Any]:
        """固定Channel數量的多模型並行視頻基準測試"""
        # 記錄測試開始時間
        test_start_time = time.time()
        
        print(f"🎬 開始固定Channel多模型並行視頻基準測試")
        print(f"   • 視頻: {video_path}")
        print(f"   • 持續時間: {duration_seconds}秒")
        print(f"   • 請求Channel數: {requested_channels}")
        
        # 驗證視頻文件
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"視頻文件不存在: {video_path}")
        
        # 獲取視頻信息
        video_info = self._get_video_info(video_path)
        print(f"📹 視頻信息: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS")
        
        # 確定要載入的模型數量
        if fixed_models is not None:
            # 使用用戶指定的固定模型數量
            max_models = fixed_models
            print(f"\n🔧 使用固定模型數量: {max_models}")
            print(f"   • 跳過自動計算，直接載入 {max_models} 個模型")
            
            # 測試載入指定數量的模型
            success, load_times, memory_usage = self._test_model_loading(max_models)
            if not success:
                print(f"❌ 無法載入 {max_models} 個模型，測試終止")
                return {}
            
            load_info = {
                'load_times': load_times,
                'memory_usage': memory_usage,
                'avg_load_time': float(np.mean(load_times)) if load_times else 0.0,
                'avg_memory_usage': float(np.mean(memory_usage)) if memory_usage else 0.0,
                'total_memory_usage': float(np.sum(memory_usage)) if memory_usage else 0.0
            }
        else:
            # 使用自動計算的模型數量
            max_models, load_info = self._find_max_loadable_models(requested_channels)
        
        if max_models == 0:
            print("❌ 無法載入任何模型，測試終止")
            return {}
        
        print(f"\n🎯 模型分配策略:")
        print(f"   • 請求Channel數: {requested_channels}")
        print(f"   • 載入模型數: {max_models}")
        
        if fixed_models is not None:
            print(f"   • 配置方式: 用戶指定固定模型數量")
        else:
            print(f"   • 配置方式: 自動計算模型數量")
        
        if max_models >= requested_channels:
            print(f"   • 分配策略: 每個Channel都有專屬模型 (理想配置)")
            channels_per_model = 1.0
        else:
            print(f"   • 分配策略: {max_models}個模型處理{requested_channels}個Channel")
            channels_per_model = requested_channels / max_models
            print(f"   • 每個模型處理: {channels_per_model:.1f}個Channel")
            
            # 解釋為什麼限制模型數量
            if fixed_models is not None:
                print(f"   • 原因: 用戶指定固定模型數量")
            elif requested_channels > 8:
                print(f"   • 原因: 避免GPU運算瓶頸，確保最佳性能")
        
        # 初始化Channel指標收集器
        channel_metrics = [FixedChannelMetric(i) for i in range(requested_channels)]
        
        # 載入模型實例
        print(f"\n🔄 載入 {max_models} 個模型實例...")
        models = []
        model_load_times = []
        model_memory_usage = []
        
        for i in range(max_models):
            model, load_time, memory_usage = self._create_model_instance(i)
            models.append(model)
            model_load_times.append(load_time)
            model_memory_usage.append(memory_usage)
        
        print(f"✅ 所有模型載入完成，總耗時: {sum(model_load_times):.3f}秒")
        
        # --- 執行一次微觀剖析 (任務 A) ---
        micro_profiling_results = {}
        if models:
            micro_profiling_results = self._profile_model_once(models[0])
        else:
            print("   ⚠️ 沒有載入任何模型，跳過微觀剖析。")
            
        # 創建Channel分配映射
        channel_to_model = {}
        for channel_id in range(requested_channels):
            if max_models >= requested_channels:
                # 理想配置：每個Channel都有專屬模型
                model_id = channel_id
                channel_metrics[channel_id].model_shared = False
            else:
                # 模型共享：多個Channel共享模型
                model_id = channel_id % max_models
                channel_metrics[channel_id].model_shared = True
            
            channel_to_model[channel_id] = model_id
            channel_metrics[channel_id].assigned_model_id = model_id
        
        print(f"📋 Channel分配映射:")
        for channel_id in range(requested_channels):
            model_id = channel_to_model[channel_id]
            print(f"   • Channel {channel_id} → Model {model_id}")
        
        # 初始化並啟動資源監控器
        resource_monitor = ResourceMonitor()
        resource_monitor.start()

        # 啟動Channel工作線程
        threads = []
        stop_ts = time.time() + duration_seconds
        
        print(f"\n🚀 啟動 {requested_channels} 個Channel工作線程...")
        
        for channel_id in range(requested_channels):
            model_id = channel_to_model[channel_id]
            thread = threading.Thread(
                target=self._fixed_channel_worker_thread,
                args=(channel_id, model_id, video_path, stop_ts, channel_metrics[channel_id], models[model_id]),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        print("✅ 開始固定Channel性能監控\n")
        
        # 定期報告
        self._fixed_channel_monitor_progress(channel_metrics, stop_ts)

        # 等待所有線程完成
        for thread in threads:
            thread.join()

        # 停止資源監控並獲取數據
        resource_monitor.stop()
        resource_monitor.join()
        resource_stats = resource_monitor.get_stats()
        
        # 清理模型
        for model in models:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 計算總執行時間
        test_end_time = time.time()
        total_execution_time = test_end_time - test_start_time
        
        # --- 👇 這裡是修正點 #1 (失誤 #1) --- 👇
        # 生成報告 (補上完整的 config 字典)
        config = {
            'model': self.model_name,
            'video': video_path,
            'requested_channels': requested_channels,
            'actual_models': max_models,
            'channels_per_model': channels_per_model,
            'fixed_models': fixed_models,
            'img_size': self.img_size,
            'video_resolution': f"{video_info['width']}x{video_info['height']}",
            'video_fps': video_info['fps'],
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'seconds': duration_seconds,
            'device': self.device,
            'model_load_time': sum(model_load_times),
            'total_execution_time': total_execution_time,
            'architecture': 'fixed_channel_multi_model_parallel',
            'hardware_specs': self.hardware_specs,
            'load_info': load_info,
            'channel_allocation': channel_to_model
        }
        
        # 將微觀剖析結果 (micro_profiling_results) 傳遞給報告生成器
        report = self._generate_fixed_channel_report(
            channel_metrics, config, resource_stats, micro_profiling_results
        )
        # --- 👆 修正結束 --- 👆
        
        self._print_fixed_channel_report(report)
        
        # 自動生成報告檔案名（如果沒有指定）
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.splitext(os.path.basename(self.model_name))[0]
            output_file = f"reports/cv_benchmark_{model_name}_{requested_channels}ch_{timestamp}.json"
        
        # 確保 reports 目錄存在
        os.makedirs("reports", exist_ok=True)
        
        # 保存報告
        self._save_report(report, output_file)
        print(f"\n📄 詳細報告已保存至: {output_file}")
        
        return report

    def run_auto_optimization(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        自動優化主函數，迭代不同的模型數量配置，執行測試，並生成最佳化報告。
        """
        # 記錄優化測試開始時間
        optimization_start_time = time.time()
        
        print(f"🚀 開始自動優化模型數量測試")
        print(f"   • 視頻: {args.video}")
        print(f"   • 持續時間: {args.seconds}秒")
        print(f"   • 請求Channel數: {args.channels}")
        
        # 獲取視頻信息
        video_info = self._get_video_info(args.video)
        print(f"📹 視頻信息: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS")
        
        # 創建唯一的報告目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_base = os.path.splitext(os.path.basename(self.model_name))[0]
        
        # 檢查是否有指定的 output_dir
        output_dir = getattr(args, 'output_dir', 'reports')
        
        report_dir_name = f"cv_optimization_{model_name_base}_{args.channels}ch_{timestamp}"
        report_dir = os.path.join(output_dir, report_dir_name)
        os.makedirs(report_dir, exist_ok=True)
        print(f"📂 報告將儲存於: {report_dir}")
        
        # 迭代測試邏輯
        test_results = []
        test_configs = list(range(1, args.channels + 1))
        
        print(f"\n🔍 將執行 {len(test_configs)} 次測試，模型數量從 1 到 {args.channels}")
        
        for i, model_count in enumerate(test_configs, 1):
            print(f"\n{'='*60}")
            print(f"🧪 測試 {i}/{len(test_configs)}: 使用 {model_count} 個模型")
            print(f"{'='*60}")
            
            try:
                # 為中間報告生成檔案路徑
                intermediate_report_name = f"benchmark_{model_count}_models.json"
                intermediate_output_file = os.path.join(report_dir, intermediate_report_name)
                
                # 執行單次測試
                result = self.benchmark_video_fixed_channels(
                    video_path=args.video,
                    duration_seconds=args.seconds,
                    requested_channels=args.channels,
                    fixed_models=model_count,
                    output_file=intermediate_output_file
                )
                
                if result and 'performance_metrics' in result:
                    perf = result['performance_metrics']
                    config = result['configuration']
                    resource_usage = perf.get('resource_usage', {})
                    
                    # 提取關鍵指標
                    avg_fps = perf['fps']['average']
                    total_fps = perf['fps']['total']
                    avg_latency = perf['latency_ms']['average']
                    channels_per_model = config['channels_per_model']
                    
                    # 計算效率分數
                    efficiency_score = self._calculate_efficiency_score(
                        avg_fps, total_fps, avg_latency,
                        args.channels, model_count, channels_per_model
                    )
                    
                    summary = {
                        'model_count': model_count,
                        'avg_fps': avg_fps,
                        'total_fps': total_fps,
                        'avg_latency': avg_latency,
                        'channels_per_model': channels_per_model,
                        'efficiency_score': efficiency_score,
                        'is_ideal_config': model_count >= args.channels,
                        'resource_usage': resource_usage,
                        'report_file': intermediate_report_name
                    }
                    
                    # 如果存在 profiling_details，則將其複製到摘要中
                    if 'profiling_details' in perf:
                        summary['profiling_details'] = perf['profiling_details']
                    
                    test_results.append(summary)
                    
                    print(f"✅ 測試完成: {model_count}個模型")
                    print(f"   • 平均FPS: {avg_fps:.2f}")
                    print(f"   • 總FPS: {total_fps:.2f}")
                    print(f"   • 平均延遲: {avg_latency:.2f}ms")
                    print(f"   • 效率分數: {efficiency_score:.2f}")
                    
                else:
                    print(f"❌ 測試失敗: {model_count}個模型")
                    
            except Exception as e:
                print(f"❌ 測試錯誤: {model_count}個模型 - {e}")
                continue
        
        # 分析結果並找到最佳配置
        if not test_results:
            print("❌ 所有測試都失敗了，無法生成優化報告")
            return {}
        
        best_config = self._find_best_configuration(test_results, args.channels)
        
        # 生成優化報告
        optimization_report = self._generate_optimization_report(
            test_results, best_config, video_info, args.channels
        )
        
        # 顯示優化結果
        self._print_optimization_report(optimization_report)
        
        # 將最終優化報告儲存到專屬資料夾中
        final_report_name = "optimization_report.json"
        final_output_file = os.path.join(report_dir, final_report_name)
        
        # 保存報告
        self._save_report(optimization_report, final_output_file)
        print(f"\n📄 優化報告已保存至: {final_output_file}")
        
        return optimization_report

    def _calculate_efficiency_score(self, avg_fps: float, total_fps: float, 
                                  avg_latency: float, requested_channels: int, 
                                  model_count: int, channels_per_model: float) -> float:
        """計算效率分數"""
        # 權重配置
        fps_weight = 0.4      # FPS權重
        latency_weight = 0.3  # 延遲權重
        efficiency_weight = 0.3  # 效率權重
        
        # FPS分數 (0-100)
        fps_score = min(100, (avg_fps / 30) * 100)  # 以30 FPS為滿分
        
        # 延遲分數 (0-100，延遲越低分數越高)
        latency_score = max(0, 100 - (avg_latency / 100) * 100)  # 以100ms為基準
        
        # 效率分數 (0-100，模型利用率越高分數越高)
        if channels_per_model >= 1.0:
            efficiency_score = 100  # 理想配置
        else:
            efficiency_score = channels_per_model * 100  # 共享模型效率
        
        # 計算總分
        total_score = (fps_score * fps_weight + 
                      latency_score * latency_weight + 
                      efficiency_score * efficiency_weight)
        
        return total_score

    def _find_best_configuration(self, results: List[Dict], requested_channels: int) -> Dict:
        """找到最佳配置"""
        if not results:
            return {}
        
        # 按效率分數排序
        sorted_results = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
        
        # 找到最佳配置
        best = sorted_results[0]
        
        # 分析配置類型
        if best['is_ideal_config']:
            config_type = "理想配置"
            recommendation = "每個Channel都有專屬模型，性能最佳"
        elif best['channels_per_model'] >= 2.0:
            config_type = "高效共享"
            recommendation = "模型共享效率高，適合高吞吐量應用"
        else:
            config_type = "平衡配置"
            recommendation = "FPS和延遲的平衡點，適合大多數應用"
        
        best['config_type'] = config_type
        best['recommendation'] = recommendation
        
        return best

    def _generate_optimization_report(self, results: List[Dict], best_config: Dict, 
                                    video_info: Dict, requested_channels: int) -> Dict:
        """生成優化報告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "sdk_info": {
                "name": "Auto-Optimization Multi-Model Benchmark",
                "version": "1.0.0",
                "framework": "PyTorch + Ultralytics YOLO"
            },
            "test_configuration": {
                "video": video_info,
                "requested_channels": requested_channels,
                "model": self.model_name,
                "device": self.device,
                "img_size": self.img_size
            },
            "test_results": results,
            "best_configuration": best_config,
            "optimization_summary": {
                "total_tests": len(results),
                "best_model_count": best_config.get('model_count', 0),
                "best_avg_fps": best_config.get('avg_fps', 0),
                "best_total_fps": best_config.get('total_fps', 0),
                "best_latency": best_config.get('avg_latency', 0),
                "efficiency_score": best_config.get('efficiency_score', 0),
                "config_type": best_config.get('config_type', ''),
                "recommendation": best_config.get('recommendation', '')
            }
        }
        
        return report

    def _print_optimization_report(self, report: Dict):
        """顯示優化報告"""
        print(f"\n{'='*80}")
        print(f"🎯 自動優化結果報告")
        print(f"{'='*80}")
        
        # 最佳配置
        best = report['best_configuration']
        summary = report['optimization_summary']
        test_config = report['test_configuration']
        
        print(f"\n🏆 最佳配置:")
        print(f"  • 模型數量: {summary['best_model_count']}")
        print(f"  • 配置類型: {summary['config_type']}")
        print(f"  • 平均FPS: {summary['best_avg_fps']:.2f}")
        print(f"  • 總FPS: {summary['best_total_fps']:.2f}")
        print(f"  • 平均延遲: {summary['best_latency']:.2f}ms")
        print(f"  • 效率分數: {summary['efficiency_score']:.2f}/100")
        print(f"  • 建議: {summary['recommendation']}")
        
        # 所有測試結果
        print(f"\n📊 所有測試結果:")
        print(f"{'模型數':<8} {'平均FPS':<10} {'總FPS':<10} {'延遲(ms)':<12} {'Avg CPU(%)':<12} {'Avg GPU(%)':<12} {'效率分數':<10} {'配置類型'}")
        print(f"{'-'*95}")
        
        for result in report['test_results']:
            config_type = "理想" if result['is_ideal_config'] else "共享"
            resource_usage = result.get('resource_usage', {})
            avg_cpu = resource_usage.get('cpu', {}).get('average', 0.0)
            avg_gpu = resource_usage.get('gpu', {}).get('average', 0.0)
            
            print(f"{result['model_count']:<8} {result['avg_fps']:<10.2f} {result['total_fps']:<10.2f} "
                  f"{result['avg_latency']:<12.2f} {avg_cpu:<12.1f} {avg_gpu:<12.1f} "
                  f"{result['efficiency_score']:<10.2f} {config_type}")
        
        # 使用建議
        print(f"\n💡 使用建議:")
        print(f"  • 最佳指令: python fixed_channel_benchmark.py --video {test_config['video']['width']}x{test_config['video']['height']} --model {self.model_name} -n {test_config['requested_channels']} -m {summary['best_model_count']} -t 30")
        print(f"  • 預期性能: 每個Channel約{summary['best_avg_fps']:.1f} FPS")
        print(f"  • 總吞吐量: {summary['best_total_fps']:.1f} frames/sec")

    def _fixed_channel_worker_thread(self,
                                   channel_id: int,
                                   model_id: int,
                                   video_path: str,
                                   stop_ts: float,
                                   metric: FixedChannelMetric,
                                   model: YOLO):
        """固定Channel工作線程函數（生產者-消費者模式）"""
        print(f"🔄 Channel {channel_id} 開始工作 (使用Model {model_id})")
        
        frame_queue = queue.Queue(maxsize=10)
        
        # --- 生產者執行緒 ---
        class ProducerThread(threading.Thread):
            # ... (生產者程式碼保持不變) ...
            def __init__(self, video_path, queue, stop_ts):
                super().__init__()
                self.daemon = True
                self.video_path = video_path
                self.queue = queue
                self.stop_ts = stop_ts
                self.read_times = []
                self.put_q_times = []
                self._stop_event = threading.Event()

            def run(self):
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    print(f"[Producer-{channel_id}] 無法打開視頻: {self.video_path}")
                    return
                
                try:
                    while time.time() < self.stop_ts and not self._stop_event.is_set():
                        t_read_start = perf_counter()
                        ret, frame = cap.read()
                        t_read_end = perf_counter()
                        self.read_times.append(t_read_end - t_read_start)

                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        
                        t_put_start = perf_counter()
                        self.queue.put(frame)
                        t_put_end = perf_counter()
                        self.put_q_times.append(t_put_end - t_put_start)
                finally:
                    cap.release()
                    # 發送結束信號
                    self.queue.put(None)

            def stop(self):
                self._stop_event.set()

        # --- 消費者邏輯 ---
        consumer_get_q_times = []
        consumer_predict_times = []

        producer = ProducerThread(video_path, frame_queue, stop_ts)
        producer.start()
        
        try:
            while True:
                t_get_start = perf_counter()
                frame = frame_queue.get()
                t_get_end = perf_counter()
                consumer_get_q_times.append(t_get_end - t_get_start)

                if frame is None:
                    break # 生產者已結束

                # --- 👇 這裡是修改重點 --- 👇
                # 接收 2 個返回值 (detections, proc_time_s)
                detections, proc_time_s = self.predict_single_frame(model, frame)
                
                # 儲存總牆上時間 (wall_s)
                consumer_predict_times.append({
                    'wall_s': proc_time_s
                })
                # --- 👆 修改結束 --- 👆
                
                metric.update(proc_time_s, len(detections))
                
        except Exception as e:
            print(f"[Channel {channel_id}] 消費者錯誤: {e}")
        finally:
            producer.stop()
            producer.join()
            
            # 回傳剖析數據
            metric.profiling_data = {
                'producer_read_times': producer.read_times,
                'producer_put_q_times': producer.put_q_times,
                'consumer_get_q_times': consumer_get_q_times,
                'consumer_predict_times': consumer_predict_times
            }

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """獲取視頻信息"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'width': 0, 'height': 0, 'fps': 0, 'frame_count': 0}
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count
        }

    def _fixed_channel_monitor_progress(self, metrics: List[FixedChannelMetric], stop_ts: float):
        """固定Channel進度監控"""
        last_emit_s = -1
        
        while time.time() < stop_ts:
            elapsed = int(stop_ts - time.time())
            now_s = int(time.time())
            
            if last_emit_s == -1 or (now_s - last_emit_s) >= 3:
                last_emit_s = now_s
                
                for metric in metrics:
                    fps = metric.get_fps()
                    latency = metric.get_latency_ms()
                    throughput = metric.get_throughput()
                    detections = metric.get_avg_detections()
                    
                    model_info = f"Model {metric.assigned_model_id}"
                    
                    print(
                        f"Channel {metric.channel_id} ({model_info}): fps={fps:.3f}, latency={latency:.2f}ms, "
                        f"detections={detections:.1f}"
                    )
                print("")
            
            time.sleep(0.5)

    def _generate_fixed_channel_report(self, 
                                     metrics: List[FixedChannelMetric], 
                                     config: Dict, 
                                     resource_stats: Dict, 
                                     micro_profiling: Dict[str, float]) -> Dict[str, Any]:
        """生成固定Channel報告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "sdk_info": {
                "name": "Fixed Channel Multi-Model Parallel Benchmark",
                "version": "1.0.0",
                "framework": "PyTorch + Ultralytics YOLO"
            },
            "configuration": config,
            "summary": {
                "total_channels": len(metrics),
                "total_models": config['actual_models'],
                "channels_per_model": config['channels_per_model'],
                "total_frames": int(sum(m.num_frames for m in metrics)),
                "total_runtime": max(m.start_time for m in metrics) - min(m.start_time for m in metrics) if metrics else 0,
                "model_load_time": config.get('model_load_time', 0)
            },
            "performance_metrics": {},
            "hardware_analysis": {},
            "optimization_recommendations": {}
        }
        
        # --- 👇 這裡是修正點 #2 (失誤 #2) --- 👇
        # 性能指標 (補上完整的 performance_metrics 字典)
        if metrics:
            fps_values = [m.get_fps() for m in metrics if m.get_fps() > 0]
            latency_values = [m.get_latency_ms() for m in metrics if m.get_latency_ms() > 0]
            throughput_values = [m.get_throughput() for m in metrics if m.get_throughput() > 0]
            
            report["performance_metrics"] = {
                "fps": {
                    "average": float(np.mean(fps_values)) if fps_values else 0.0,
                    "min": float(np.min(fps_values)) if fps_values else 0.0,
                    "max": float(np.max(fps_values)) if fps_values else 0.0,
                    "per_channel": fps_values,
                    "total": float(np.sum(fps_values)) if fps_values else 0.0
                },
                "latency_ms": {
                    "average": float(np.mean(latency_values)) if latency_values else 0.0,
                    "min": float(np.min(latency_values)) if latency_values else 0.0,
                    "max": float(np.max(latency_values)) if latency_values else 0.0,
                    "per_channel": latency_values
                },
                "throughput": {
                    "total": float(np.sum(throughput_values)) if throughput_values else 0.0,
                    "per_channel": throughput_values
                },
                "resource_usage": resource_stats
            }
        # --- 👆 修正結束 --- 👆

        # 微觀性能剖析 (合併 宏觀實測值(B) 和 微觀理論值(A))
        profiling_details = {}
        if metrics:
            for m in metrics:
                if m.profiling_data:
                    def _avg_ms(data, key=None):
                        if not data:
                            return 0.0
                        values = [d.get(key, 0) for d in data] if key else data
                        return (sum(values) / len(values)) * 1000 if values else 0.0

                    predict_times = m.profiling_data.get('consumer_predict_times', [])
                    
                    # 1. 獲取宏觀實測數據 (Task B)
                    macro_data = {
                        "macro_producer_read_avg_ms": _avg_ms(m.profiling_data.get('producer_read_times', [])),
                        "macro_producer_put_q_avg_ms": _avg_ms(m.profiling_data.get('producer_put_q_times', [])),
                        "macro_consumer_get_q_avg_ms": _avg_ms(m.profiling_data.get('consumer_get_q_times', [])),
                        "macro_consumer_wall_avg_ms": _avg_ms(predict_times, key='wall_s'), # 這是總延遲
                    }
                    
                    # 2. 存儲宏觀數據
                    profiling_details[f"channel_{m.channel_id}"] = macro_data
                    
                    # 3. 併入微觀理論數據 (Task A)
                    # (micro_profiling 是從 benchmark_video_fixed_channels 傳入的)
                    profiling_details[f"channel_{m.channel_id}"].update(micro_profiling)

        
        # 將剖析數據加入到 performance_metrics 中
        if profiling_details:
            report["performance_metrics"]["profiling_details"] = profiling_details
        
        # 硬體分析
        report["hardware_analysis"] = {
            "hardware_specs": self.hardware_specs,
            "channel_allocation": {
                "requested_channels": config['requested_channels'],
                "actual_models": config['actual_models'],
                "channels_per_model": config['channels_per_model'],
                "allocation_efficiency": min(100.0, config['actual_models'] / config['requested_channels'] * 100),
                "is_ideal_config": config['actual_models'] >= config['requested_channels']
            },
            "memory_utilization": {
                # "estimated_model_memory": 0, # 已棄用
                "total_used_memory": config.get('load_info', {}).get('total_memory_usage', 0),
                "available_memory": self.hardware_specs.get('gpu_memory_gb', 0) if self.device != 'cpu' else self.hardware_specs.get('total_memory_gb', 0)
            }
        }
        
        # 優化建議
        recommendations = []
        
        if config['actual_models'] >= config['requested_channels']:
            recommendations.append("✅ 理想配置：每個Channel都有專屬模型")
            recommendations.append("✅ 性能最佳：無模型共享，無資源競爭")
        else:
            recommendations.append(f"⚠️ 模型共享：每個模型處理 {config['channels_per_model']:.1f} 個Channel")
            recommendations.append(f"⚠️ 硬體限制：只能載入 {config['actual_models']}/{config['requested_channels']} 個模型")
            
            if config['requested_channels'] > 8:
                recommendations.append("💡 原因：GPU運算瓶頸，避免過多模型同時運行")
                recommendations.append("💡 建議：使用更小的模型 (yolov8n) 以載入更多實例")
                recommendations.append("💡 建議：考慮使用批次處理來提升效率")
            else:
                if self.device != 'cpu':
                    recommendations.append("💡 建議：升級GPU記憶體以支援更多模型")
                    recommendations.append("💡 建議：使用更小的模型 (yolov8n) 以載入更多實例")
                else:
                    recommendations.append("💡 建議：增加系統記憶體或使用GPU加速")
        
        report["optimization_recommendations"] = recommendations
        
        return report

    def _print_fixed_channel_report(self, report: Dict[str, Any]):
        """打印固定Channel報告"""
        print("\n" + "="*80)
        print("🚀 Innodisk Computer Vision Benchmark 測試報告 v1.0")
        print("="*80)
        
        # 測試配置
        config = report["configuration"]
        print(f"\n📊 測試配置:")
        print(f"  • 模型: {config['model']}")
        print(f"  • 視頻: {config['video']}")
        print(f"  • 請求Channel數: {config['requested_channels']}")
        print(f"  • 實際模型數: {config['actual_models']}")
        print(f"  • 每模型處理Channel數: {config['channels_per_model']:.1f}")
        if config.get('fixed_models') is not None:
            print(f"  • 固定模型數量: {config['fixed_models']} (用戶指定)")
        print(f"  • 模型載入時間: {config['model_load_time']:.3f}秒")
        print(f"  • 總執行時間: {config['total_execution_time']:.3f}秒")
        print(f"  • 模型輸入尺寸: {config['img_size']}x{config['img_size']}")
        print(f"  • 視頻解析度: {config['video_resolution']}")
        print(f"  • 視頻FPS: {config['video_fps']:.2f}")
        print(f"  • 置信度閾值: {config['conf']}")
        print(f"  • IoU閾值: {config['iou']}")
        print(f"  • 測試持續時間: {config['seconds']}秒")
        print(f"  • 設備: {config['device']}")
        
        # 硬體規格
        hw_specs = config['hardware_specs']
        print(f"\n💻 硬體規格:")
        print(f"  • CPU核心數: {hw_specs['cpu_cores']}")
        print(f"  • CPU線程數: {hw_specs['cpu_threads']}")
        print(f"  • 系統記憶體: {hw_specs['total_memory_gb']:.1f} GB")
        if hw_specs['cuda_available']:
            print(f"  • GPU數量: {hw_specs['gpu_count']}")
            if hw_specs['gpu_count'] > 1:
                # 多GPU環境：分別顯示每個GPU
                for gpu in hw_specs['gpus']:
                    print(f"  • GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB, Compute {gpu['compute_capability']})")
            else:
                # 單GPU環境：保持原有格式
                print(f"  • GPU記憶體: {hw_specs['gpu_memory_gb']:.1f} GB")
                print(f"  • GPU名稱: {hw_specs['gpu_name']}")
        
        # 性能指標
        if "performance_metrics" in report and report["performance_metrics"]:
            perf = report["performance_metrics"]
            print(f"\n⚡ 性能指標:")
            print(f"  • 平均每Channel FPS: {perf['fps']['average']:.2f}")
            print(f"  • FPS範圍: {perf['fps']['min']:.2f} - {perf['fps']['max']:.2f}")
            print(f"  • 總FPS (所有Channel合計): {perf['fps']['total']:.2f}")
            print(f"  • 平均延遲: {perf['latency_ms']['average']:.2f}ms")
            print(f"  • 延遲範圍: {perf['latency_ms']['min']:.2f} - {perf['latency_ms']['max']:.2f}ms")
            
            # 資源使用率統計
            if "resource_usage" in perf:
                res = perf["resource_usage"]
                print(f"\n💻 資源使用率:")
                print(f"  • CPU使用率: 平均 {res['cpu']['average']:.1f}% (範圍: {res['cpu']['min']:.1f}% - {res['cpu']['max']:.1f}%)")
                print(f"  • 記憶體使用率: 平均 {res['memory']['average']:.1f}% (範圍: {res['memory']['min']:.1f}% - {res['memory']['max']:.1f}%)")
                print(f"  • GPU使用率: 平均 {res['gpu']['average']:.1f}% (範圍: {res['gpu']['min']:.1f}% - {res['gpu']['max']:.1f}%)")
        
        # 硬體分析
        hw_analysis = report["hardware_analysis"]
        print(f"\n🔍 硬體分析:")
        print(f"  • Channel分配: {hw_analysis['channel_allocation']['requested_channels']} → {hw_analysis['channel_allocation']['actual_models']} 模型")
        print(f"  • 每模型處理: {hw_analysis['channel_allocation']['channels_per_model']:.1f} 個Channel")
        print(f"  • 分配效率: {hw_analysis['channel_allocation']['allocation_efficiency']:.1f}%")
        
        if hw_analysis['channel_allocation']['is_ideal_config']:
            print(f"  • 配置狀態: ✅ 理想配置 (每個Channel都有專屬模型)")
        else:
            print(f"  • 配置狀態: ⚠️ 模型共享 (多個Channel共享模型)")
        
        # print(f"  • 估算模型記憶體: {hw_analysis['memory_utilization']['estimated_model_memory']:.1f} GB") # 已棄用
        print(f"  • 總使用記憶體: {hw_analysis['memory_utilization']['total_used_memory']:.1f} GB")
        print(f"  • 可用記憶體: {hw_analysis['memory_utilization']['available_memory']:.1f} GB")
        
        # 優化建議
        if report["optimization_recommendations"]:
            print(f"\n💡 優化建議:")
            for i, recommendation in enumerate(report["optimization_recommendations"], 1):
                print(f"  {i}. {recommendation}")
        
        # 效率分數計算說明
        print(f"\n📊 效率分數計算說明:")
        print(f"  效率分數是一個綜合評分系統 (總分100分)，用來評估不同模型配置的整體性能表現：")
        print(f"  • FPS分數 (40%權重): 以30 FPS為滿分，計算公式: min(100, (平均FPS/30) × 100)")
        print(f"  • 延遲分數 (30%權重): 以100ms為基準，延遲越低分數越高，計算公式: max(0, 100 - (平均延遲/100) × 100)")
        print(f"  • 效率分數 (30%權重): 理想配置(每Channel專屬模型)為100分，共享模型為 channels_per_model × 100")
        print(f"  • 總分計算: FPS分數×0.4 + 延遲分數×0.3 + 效率分數×0.3")
        print(f"  • 分數意義: 0-30分(需優化) | 30-60分(可接受) | 60-80分(良好) | 80-100分(優秀)")
        
        print("\n" + "="*80)

    def _save_report(self, report: Dict[str, Any], output_file: str):
        """保存報告到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="固定Channel數量的多模型並行基準測試工具")
    parser.add_argument("--video", type=str, required=True, help="視頻文件路徑")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 模型名稱或路徑")
    parser.add_argument("-n", "--channels", type=int, default=4, help="固定的並行Channel數（不會改變）")
    parser.add_argument("-m", "--models", type=int, help="固定載入的模型數量（覆蓋自動計算）")
    parser.add_argument("--auto-optimize", action="store_true", help="自動測試從1到N個模型數量，找到最佳平衡點")
    parser.add_argument("-t", "--seconds", type=int, default=60, help="測試持續時間（秒）")
    parser.add_argument("--img-size", type=int, default=640, help="模型輸入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度閾值")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 閾值")
    parser.add_argument("--device", type=str, default="cuda", help="設備配置 (auto, cpu, cuda)")
    parser.add_argument("--output", type=str, help="輸出報告文件路徑 (單次測試) 或報告目錄 (自動優化)")
    
    args = parser.parse_args()
    
    # 將 output 參數作為 output_dir 傳遞給自動優化
    if args.output:
        args.output_dir = args.output
    else:
        args.output_dir = "reports"

    try:
        # 創建固定Channel基準測試器
        benchmark = FixedChannelBenchmark(
            model_name=args.model,
            device=args.device,
            img_size=args.img_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # 執行基準測試
        if args.auto_optimize:
            # 自動優化模式：測試從1到N個模型數量
            report = benchmark.run_auto_optimization(args)
        else:
            # 單次測試模式
            report = benchmark.benchmark_video_fixed_channels(
                video_path=args.video,
                duration_seconds=args.seconds,
                requested_channels=args.channels,
                fixed_models=args.models,
                output_file=args.output
            )
        
        print("\n✅ 基準測試完成！")
        
    except Exception as e:
        print(f"❌ 基準測試失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
