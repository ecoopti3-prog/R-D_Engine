"""
data_movement.py — Memory wall and bandwidth physics validators.
"""
from __future__ import annotations
import math
from typing import Tuple

# ── Physical limits ───────────────────────────────────────────────────────────
SPEED_OF_LIGHT_CM_PER_NS = 30.0   # ~c in dielectric (ε_r ≈ 4 → ~15 cm/ns, use 30 as upper)
DRAM_LATENCY_MIN_NS       = 10.0  # ~10ns minimum for DRAM access (physics of charge)
HBM3_MAX_BW_GB_S          = 1200.0  # HBM3 peak bandwidth
PCIE5_MAX_BW_GB_S         = 128.0   # PCIe 5.0 x16
NVLINK4_MAX_BW_GB_S       = 900.0   # NVLink 4.0 total


def check_bandwidth_wall(bandwidth_gb_s: float, compute_tflops: float) -> Tuple[bool, str]:
    """
    Roofline model check: is the system compute-bound or memory-bound?
    Arithmetic Intensity threshold = Compute / Bandwidth (FLOP/byte)
    """
    if bandwidth_gb_s <= 0 or compute_tflops <= 0:
        return False, f"Invalid values: bandwidth={bandwidth_gb_s}, compute={compute_tflops}"
    arithmetic_intensity = (compute_tflops * 1e12) / (bandwidth_gb_s * 1e9)   # FLOP/byte
    ridge_point = compute_tflops * 1000 / bandwidth_gb_s   # simplified ridge point
    status = "compute-bound" if arithmetic_intensity > ridge_point else "memory-bound"
    # Flag if bandwidth exceeds any known standard
    if bandwidth_gb_s > HBM3_MAX_BW_GB_S * 4:   # 4× HBM3 stacks
        return (
            False,
            f"Claimed bandwidth {bandwidth_gb_s} GB/s exceeds 4×HBM3 limit "
            f"{HBM3_MAX_BW_GB_S * 4} GB/s — not achievable with current technology"
        )
    return True, f"System is {status} (AI={arithmetic_intensity:.0f} FLOP/byte) — OK"


def check_memory_latency(latency_ns: float, access_type: str = "dram") -> Tuple[bool, str]:
    """
    Validate claimed memory latency against physical signal propagation limits.
    """
    if access_type.lower() == "dram":
        if latency_ns < DRAM_LATENCY_MIN_NS:
            return (
                False,
                f"Claimed DRAM latency {latency_ns}ns below physical minimum "
                f"{DRAM_LATENCY_MIN_NS}ns (charge physics)"
            )
    # SRAM: RC delay limits ~0.1-1ns per stage
    elif access_type.lower() == "sram":
        SRAM_MIN_NS = 0.1
        if latency_ns < SRAM_MIN_NS:
            return False, f"Claimed SRAM latency {latency_ns}ns below RC limit {SRAM_MIN_NS}ns"
    return True, f"{access_type.upper()} latency {latency_ns}ns within physical limits — OK"


def check_interconnect(bandwidth_gb_s: float, distance_mm: float) -> Tuple[bool, str]:
    """
    Validate interconnect bandwidth vs physical signal propagation.
    At high frequency, signal integrity limits bandwidth over distance.
    """
    # Shannon–Hartley rough estimate: max throughput ~ 1 bit per mm at >1Tbps rates
    # Practical: PCIe5 retimers at >200mm degrade signal
    if distance_mm > 300 and bandwidth_gb_s > PCIE5_MAX_BW_GB_S:
        return (
            False,
            f"Bandwidth {bandwidth_gb_s} GB/s over {distance_mm}mm — "
            f"exceeds PCIe5 {PCIE5_MAX_BW_GB_S} GB/s practical limit at this distance"
        )
    if distance_mm > 500 and bandwidth_gb_s > 50:
        return (
            False,
            f"High bandwidth {bandwidth_gb_s} GB/s over {distance_mm}mm requires "
            f"repeater infrastructure — signal integrity physically limited"
        )
    return True, f"Interconnect {bandwidth_gb_s} GB/s over {distance_mm}mm — within limits — OK"
