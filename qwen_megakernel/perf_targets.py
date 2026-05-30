"""Performance targets for megakernel decode and voice pipeline (RTX 5090)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerfTargets:
    """SLOs for decode and voice latency."""

    decode_tok_s_min: float = 800.0
    ttfc_ms_max: float = 90.0
    rtf_max: float = 0.3
    e2e_latency_ms_max: float = 2000.0


TARGETS = PerfTargets()


def check_min(name: str, value: float, target: float, *, unit: str = "") -> bool:
    ok = value >= target
    mark = "PASS" if ok else "FAIL"
    suffix = f" {unit}" if unit else ""
    print(
        f"  {name:<22} {value:>10.2f}{suffix}  (target ≥ {target:.2f}{suffix})  [{mark}]"
    )
    return ok


def check_max(name: str, value: float, target: float, *, unit: str = "") -> bool:
    ok = value <= target
    mark = "PASS" if ok else "FAIL"
    suffix = f" {unit}" if unit else ""
    print(
        f"  {name:<22} {value:>10.2f}{suffix}  (target ≤ {target:.2f}{suffix})  [{mark}]"
    )
    return ok
