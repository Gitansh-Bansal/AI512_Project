# ------------------------------------------------------------------------------
# Task 3 Point 5 Model Complexity & Speed Benchmarking Suite
# Profiles: Number of parameters, FLOPs / MACs, Inference Latency (ms), and FPS
# ------------------------------------------------------------------------------

import os
import sys
import time
import platform
import psutil
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import thop

def get_hardware_info(device: torch.device) -> Dict[str, str]:
    """Retrieves detailed hardware specifications for benchmarking context."""
    info = {
        'OS': f"{platform.system()} {platform.release()} ({platform.version()})",
        'Python_Version': platform.python_version(),
        'PyTorch_Version': torch.__version__,
        'Device': str(device),
        'CPU_Processor': platform.processor() or platform.machine(),
        'CPU_Physical_Cores': str(psutil.cpu_count(logical=False)),
        'CPU_Logical_Threads': str(psutil.cpu_count(logical=True)),
        'RAM_Total_GB': f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    }
    if device.type == 'cuda' and torch.cuda.is_available():
        info['GPU_Device_Name'] = torch.cuda.get_device_name(0)
        info['GPU_Memory_GB'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    else:
        info['GPU_Device_Name'] = 'N/A (CPU Execution)'
        info['GPU_Memory_GB'] = 'N/A'
    return info

def profile_parameters(model: nn.Module) -> Dict[str, float]:
    """Calculates total and trainable parameter counts in Millions (M)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params_raw': total_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_raw': trainable_params,
        'trainable_params_M': trainable_params / 1e6
    }

def profile_flops(model: nn.Module, input_resolutions: Optional[List[Tuple[int, int]]] = None, device: Optional[torch.device] = None) -> List[Dict[str, object]]:
    """
    Computes FLOPs and MACs across target image resolutions.
    Default resolutions: (512, 512), (512, 1024), (1024, 2048)
    """
    if input_resolutions is None:
        input_resolutions = [(512, 512), (512, 1024), (1024, 2048)]
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    for h, w in input_resolutions:
        dummy_input = torch.randn(1, 3, h, w, device=device)
        try:
            # thop modifies model buffers; we profile on a shallow wrapper or pass verbose=False
            macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
            flops = 2.0 * macs  # Standard FLOPs estimation (1 MAC = 2 FLOPs)
            results.append({
                'Resolution': f"{w}x{h}",
                'Height': h,
                'Width': w,
                'MACs_G': macs / 1e9,
                'FLOPs_G': flops / 1e9,
                'Params_M': params / 1e6
            })
        except Exception as e:
            results.append({
                'Resolution': f"{w}x{h}",
                'Height': h,
                'Width': w,
                'Error': str(e)
            })

    return results

def benchmark_inference_speed(
    model: nn.Module,
    input_resolution: Tuple[int, int] = (512, 1024),
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    warmup_runs: int = 15,
    timed_runs: int = 50
) -> Dict[str, object]:
    """
    Measures inference latency and FPS with strict hardware synchronization.
    """
    if device is None:
        device = next(model.parameters()).device

    h, w = input_resolution
    dummy_input = torch.randn(batch_size, 3, h, w, device=device)

    model.eval()
    is_cuda = device.type == 'cuda' and torch.cuda.is_available()

    # Warmup phase to stabilize GPU clock rates / CPU caches / JIT trace
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()

    # Timed measurement phase
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # milliseconds

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))
    median_latency = float(np.median(latencies))
    fps = float((1000.0 * batch_size) / mean_latency)

    return {
        'Resolution': f"{w}x{h}",
        'Batch_Size': batch_size,
        'Device': str(device),
        'Warmup_Iterations': warmup_runs,
        'Timed_Iterations': timed_runs,
        'Mean_Latency_ms': mean_latency,
        'mean_latency_ms': mean_latency,
        'Std_Latency_ms': std_latency,
        'Min_Latency_ms': min_latency,
        'Max_Latency_ms': max_latency,
        'Median_Latency_ms': median_latency,
        'FPS': fps
    }
