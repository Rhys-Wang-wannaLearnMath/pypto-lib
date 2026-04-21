#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Performance Analyzer for PyPTO-Lib build_output.

One-click analysis of all performance data in build_output/:
  - perf_swimlane_*.json (tasks, scheduler phases, orchestrator phases)
  - kernel_config.py (func_id to kernel name mapping)
  - memory_after_AllocateMemoryAddr.txt (memory usage per kernel)

Outputs:
  - Terminal summary with colored tables
  - Structured Markdown report

Usage:
    python tools/perf_analyzer.py build_output/                    # Analyze all scopes
    python tools/perf_analyzer.py build_output/Qwen3Scope1_*/      # Single scope
    python tools/perf_analyzer.py build_output/ -o report.md       # Custom output path
    python tools/perf_analyzer.py build_output/ -v                 # Verbose mode
"""

import argparse
import importlib.util
import json
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskStats:
    func_id: int
    func_name: str
    core_type: str
    count: int = 0
    total_exec_us: float = 0.0
    total_latency_us: float = 0.0
    total_head_oh_us: float = 0.0
    total_tail_oh_us: float = 0.0
    min_exec_us: float = float("inf")
    max_exec_us: float = 0.0

    @property
    def avg_exec_us(self) -> float:
        return self.total_exec_us / self.count if self.count else 0.0

    @property
    def avg_latency_us(self) -> float:
        return self.total_latency_us / self.count if self.count else 0.0

    @property
    def avg_head_oh_us(self) -> float:
        return self.total_head_oh_us / self.count if self.count else 0.0

    @property
    def avg_tail_oh_us(self) -> float:
        return self.total_tail_oh_us / self.count if self.count else 0.0

    @property
    def exec_latency_ratio(self) -> float:
        return (self.total_exec_us / self.total_latency_us * 100) if self.total_latency_us > 0 else 0.0


@dataclass
class SchedulerStats:
    thread_id: int
    phase_counts: dict = field(default_factory=lambda: defaultdict(int))
    phase_durations_us: dict = field(default_factory=lambda: defaultdict(float))
    total_duration_us: float = 0.0

    @property
    def idle_ratio(self) -> float:
        idle = self.phase_durations_us.get("idle", 0.0)
        return (idle / self.total_duration_us * 100) if self.total_duration_us > 0 else 0.0


@dataclass
class OrchestratorStats:
    phase_counts: dict = field(default_factory=lambda: defaultdict(int))
    phase_durations_us: dict = field(default_factory=lambda: defaultdict(float))
    total_duration_us: float = 0.0
    total_tasks: int = 0

    def phase_ratio(self, phase: str) -> float:
        dur = self.phase_durations_us.get(phase, 0.0)
        return (dur / self.total_duration_us * 100) if self.total_duration_us > 0 else 0.0


@dataclass
class MemoryEntry:
    kernel_name: str
    space: str
    used_kb: float
    limit_kb: float
    usage_pct: float
    mem_refs: int


@dataclass
class ScopeAnalysis:
    name: str
    path: Path
    # Task-level
    task_count: int = 0
    total_time_us: float = 0.0
    task_stats: dict = field(default_factory=dict)  # func_id -> TaskStats
    min_dispatch_us: float = float("inf")
    max_finish_us: float = 0.0
    # Scheduler
    scheduler_stats: list = field(default_factory=list)  # list[SchedulerStats]
    # Orchestrator
    orchestrator_stats: Optional[OrchestratorStats] = None
    # Memory
    memory_entries: list = field(default_factory=list)  # list[MemoryEntry]
    # Config
    block_dim: int = 0
    aicpu_thread_num: int = 0
    runtime: str = ""
    # Kernel mapping
    func_id_to_name: dict = field(default_factory=dict)
    func_id_to_core_type: dict = field(default_factory=dict)

    @property
    def wall_time_us(self) -> float:
        if self.min_dispatch_us < float("inf") and self.max_finish_us > 0:
            return self.max_finish_us - self.min_dispatch_us
        return self.total_time_us

    @property
    def wall_time_ms(self) -> float:
        return self.wall_time_us / 1000.0


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def load_kernel_config(config_path: Path) -> dict:
    """Load kernel_config.py and return (func_id_to_name, func_id_to_core_type, runtime_config)."""
    result = {"names": {}, "core_types": {}, "runtime": {}}

    if not config_path.exists():
        return result

    spec = importlib.util.spec_from_file_location("kernel_config", config_path)
    if spec is None or spec.loader is None:
        return result

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return result

    if hasattr(module, "KERNELS"):
        for kernel in module.KERNELS:
            fid = kernel.get("func_id")
            if fid is None:
                continue
            # Extract name from source path if 'name' field not present
            name = kernel.get("name")
            if not name:
                src = kernel.get("source", "")
                name = Path(src).stem if src else f"func_{fid}"
            result["names"][fid] = name
            result["core_types"][fid] = kernel.get("core_type", "unknown")

    if hasattr(module, "RUNTIME_CONFIG"):
        result["runtime"] = module.RUNTIME_CONFIG

    return result


def parse_memory_report(report_path: Path) -> list:
    """Parse memory_after_AllocateMemoryAddr.txt into list of MemoryEntry."""
    entries = []
    if not report_path.exists():
        return entries

    text = report_path.read_text()
    current_kernel = None

    for line in text.splitlines():
        line = line.strip()
        # Match kernel header: --- kernel_name ---
        m = re.match(r"^---\s+(\S+)\s+---$", line)
        if m:
            current_kernel = m.group(1)
            continue

        if current_kernel is None:
            continue

        # Match data row: Space | Used | Limit | Usage | MemRefs
        # Example:  Vec    |    82.2 KB  |   192.0 KB  |   42.8%  |  7
        m = re.match(
            r"(\w+)\s+\|\s+([\d.]+)\s+KB\s+\|\s+([\d.]+)\s+KB\s+\|\s+([\d.]+)%\s+\|\s+(\d+)",
            line,
        )
        if m:
            entries.append(
                MemoryEntry(
                    kernel_name=current_kernel,
                    space=m.group(1),
                    used_kb=float(m.group(2)),
                    limit_kb=float(m.group(3)),
                    usage_pct=float(m.group(4)),
                    mem_refs=int(m.group(5)),
                )
            )

    return entries


def parse_perf_swimlane(swimlane_path: Path) -> dict:
    """Parse perf_swimlane_*.json and return the raw data dict."""
    with open(swimlane_path) as f:
        return json.load(f)


def _func_id_to_letter(func_id: int) -> str:
    letters = []
    m = func_id + 1
    while m > 0:
        m, rem = divmod(m - 1, 26)
        letters.append(chr(ord("a") + rem))
    return str(func_id) + "_" + "".join(reversed(letters))


def analyze_scope(scope_dir: Path, verbose: bool = False) -> Optional[ScopeAnalysis]:
    """Analyze a single scope directory and return ScopeAnalysis."""
    scope_name = scope_dir.name
    analysis = ScopeAnalysis(name=scope_name, path=scope_dir)

    # 1. Load kernel config
    config_path = scope_dir / "kernel_config.py"
    kconfig = load_kernel_config(config_path)
    analysis.func_id_to_name = kconfig["names"]
    analysis.func_id_to_core_type = kconfig["core_types"]
    rt_cfg = kconfig["runtime"]
    analysis.block_dim = rt_cfg.get("block_dim", 0)
    analysis.aicpu_thread_num = rt_cfg.get("aicpu_thread_num", 0)
    analysis.runtime = rt_cfg.get("runtime", "")

    # 2. Load memory report
    mem_path = scope_dir / "report" / "memory_after_AllocateMemoryAddr.txt"
    analysis.memory_entries = parse_memory_report(mem_path)

    # 3. Load perf swimlane
    swimlane_dir = scope_dir / "swimlane_data"
    perf_files = sorted(swimlane_dir.glob("perf_swimlane_*.json"))
    if not perf_files:
        if verbose:
            print(f"  [WARN] No perf_swimlane_*.json in {swimlane_dir}")
        return analysis

    perf_path = perf_files[-1]  # Use latest
    if verbose:
        print(f"  Loading: {perf_path.name}")

    data = parse_perf_swimlane(perf_path)
    tasks = data.get("tasks", [])
    analysis.task_count = len(tasks)

    # 3a. Task-level analysis
    for task in tasks:
        fid = task["func_id"]
        if fid not in analysis.task_stats:
            name = analysis.func_id_to_name.get(fid, f"func_{_func_id_to_letter(fid)}")
            ctype = analysis.func_id_to_core_type.get(fid, task.get("core_type", "unknown"))
            analysis.task_stats[fid] = TaskStats(func_id=fid, func_name=name, core_type=ctype)

        ts = analysis.task_stats[fid]
        ts.count += 1

        exec_us = task["duration_us"]
        ts.total_exec_us += exec_us
        ts.min_exec_us = min(ts.min_exec_us, exec_us)
        ts.max_exec_us = max(ts.max_exec_us, exec_us)

        dispatch_us = task.get("dispatch_time_us", 0)
        finish_us = task.get("finish_time_us", 0)
        start_us = task["start_time_us"]
        end_us = task["end_time_us"]

        if dispatch_us >= 0 and finish_us > 0:
            head_oh = start_us - dispatch_us
            tail_oh = finish_us - end_us
            latency = finish_us - dispatch_us
            ts.total_head_oh_us += head_oh
            ts.total_tail_oh_us += tail_oh
            ts.total_latency_us += latency
            analysis.min_dispatch_us = min(analysis.min_dispatch_us, dispatch_us)
            analysis.max_finish_us = max(analysis.max_finish_us, finish_us)

    if tasks:
        starts = [t["start_time_us"] for t in tasks]
        ends = [t["end_time_us"] for t in tasks]
        analysis.total_time_us = max(ends) - min(starts)

    # 3b. Scheduler phase analysis
    sched_phases = data.get("aicpu_scheduler_phases", [])
    for thread_idx, records in enumerate(sched_phases):
        if not records:
            continue
        ss = SchedulerStats(thread_id=thread_idx)
        for rec in records:
            phase = rec.get("phase", "unknown")
            dur = rec["end_time_us"] - rec["start_time_us"]
            ss.phase_counts[phase] += 1
            ss.phase_durations_us[phase] += dur
            ss.total_duration_us += dur
        analysis.scheduler_stats.append(ss)

    # 3c. Orchestrator phase analysis
    orch_phases = data.get("aicpu_orchestrator_phases", [])
    if orch_phases:
        os = OrchestratorStats()
        for thread_records in orch_phases:
            for rec in thread_records:
                phase = rec.get("phase", "unknown")
                dur = rec["end_time_us"] - rec["start_time_us"]
                os.phase_counts[phase] += 1
                os.phase_durations_us[phase] += dur
                os.total_duration_us += dur
                os.total_tasks = max(os.total_tasks, rec.get("submit_idx", 0) + 1)
        analysis.orchestrator_stats = os

    return analysis


def discover_scopes(base_dir: Path) -> list:
    """Discover scope directories under a base directory.

    A scope directory must contain kernel_config.py.
    If base_dir itself is a scope, returns [base_dir].
    """
    if (base_dir / "kernel_config.py").exists():
        return [base_dir]

    scopes = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and (child / "kernel_config.py").exists():
            scopes.append(child)
    return scopes


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_us(us: float) -> str:
    """Format microseconds for display."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f} s"
    if us >= 1_000:
        return f"{us / 1_000:.2f} ms"
    return f"{us:.2f} us"


def _fmt_pct(pct: float) -> str:
    return f"{pct:.1f}%"


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_scope_summary(analysis: ScopeAnalysis):
    """Print terminal summary for a single scope."""
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  SCOPE: {analysis.name}")
    print(f"  Path:  {analysis.path}")
    print(sep)

    # General info
    print(f"\n  Runtime: {analysis.runtime}  |  Block Dim: {analysis.block_dim}  |  "
          f"AICPU Threads: {analysis.aicpu_thread_num}  |  Tasks: {analysis.task_count}")
    print(f"  Wall Time: {_fmt_us(analysis.wall_time_us)}")

    # Task statistics table
    if analysis.task_stats:
        _print_task_table(analysis)

    # Scheduler stats
    if analysis.scheduler_stats:
        _print_scheduler_table(analysis)

    # Orchestrator stats
    if analysis.orchestrator_stats:
        _print_orchestrator_table(analysis)

    # Memory stats
    if analysis.memory_entries:
        _print_memory_table(analysis)

    # Diagnostics
    _print_diagnostics(analysis)


def _print_task_table(analysis: ScopeAnalysis):
    print(f"\n  --- Task Statistics by Function ---")
    print(f"  {'FuncID':<7} {'Name':<28} {'Type':<5} {'Count':>5}  "
          f"{'Avg Exec':>12}  {'Avg Latency':>12}  {'Exec%':>6}  "
          f"{'Avg HeadOH':>12}  {'Avg TailOH':>12}")
    print(f"  {'-' * 115}")

    total_exec = 0.0
    total_latency = 0.0
    total_count = 0

    for fid in sorted(analysis.task_stats.keys()):
        ts = analysis.task_stats[fid]
        total_exec += ts.total_exec_us
        total_latency += ts.total_latency_us
        total_count += ts.count

        print(f"  {fid:<7} {ts.func_name:<28} {ts.core_type:<5} {ts.count:>5}  "
              f"{_fmt_us(ts.avg_exec_us):>12}  {_fmt_us(ts.avg_latency_us):>12}  "
              f"{_fmt_pct(ts.exec_latency_ratio):>6}  "
              f"{_fmt_us(ts.avg_head_oh_us):>12}  {_fmt_us(ts.avg_tail_oh_us):>12}")

    print(f"  {'-' * 115}")
    overall_ratio = (total_exec / total_latency * 100) if total_latency > 0 else 0
    print(f"  {'TOTAL':<41} {total_count:>5}  "
          f"{_fmt_us(total_exec):>12}  {_fmt_us(total_latency):>12}  {_fmt_pct(overall_ratio):>6}")


def _print_scheduler_table(analysis: ScopeAnalysis):
    print(f"\n  --- Scheduler Thread Statistics ---")
    all_phases = set()
    for ss in analysis.scheduler_stats:
        all_phases.update(ss.phase_durations_us.keys())
    phase_order = ["complete", "dispatch", "scan", "idle"]
    phases = [p for p in phase_order if p in all_phases]
    phases += sorted(all_phases - set(phase_order))

    header = f"  {'Thread':>6}  {'Total':>10}"
    for p in phases:
        header += f"  {p:>12}"
    header += f"  {'Idle%':>6}"
    print(header)
    print(f"  {'-' * (22 + 14 * len(phases) + 8)}")

    for ss in analysis.scheduler_stats:
        row = f"  {ss.thread_id:>6}  {_fmt_us(ss.total_duration_us):>10}"
        for p in phases:
            dur = ss.phase_durations_us.get(p, 0)
            row += f"  {_fmt_us(dur):>12}"
        row += f"  {_fmt_pct(ss.idle_ratio):>6}"
        print(row)


def _print_orchestrator_table(analysis: ScopeAnalysis):
    os = analysis.orchestrator_stats
    print(f"\n  --- Orchestrator Phase Breakdown ---")
    print(f"  Total Duration: {_fmt_us(os.total_duration_us)}  |  Tasks Submitted: {os.total_tasks}")

    phase_order = ["orch_alloc", "orch_sync", "orch_lookup", "orch_insert",
                    "orch_params", "orch_fanin", "orch_scope_end"]
    all_phases = sorted(os.phase_durations_us.keys())
    phases = [p for p in phase_order if p in os.phase_durations_us]
    phases += [p for p in all_phases if p not in phase_order]

    print(f"\n  {'Phase':<20} {'Duration':>12} {'Count':>8} {'Ratio':>8}  {'Bar'}")
    print(f"  {'-' * 70}")

    for phase in phases:
        dur = os.phase_durations_us[phase]
        cnt = os.phase_counts[phase]
        ratio = os.phase_ratio(phase)
        display = phase.replace("orch_", "") if phase.startswith("orch_") else phase
        print(f"  {display:<20} {_fmt_us(dur):>12} {cnt:>8} {_fmt_pct(ratio):>8}  {_bar(ratio)}")


def _print_memory_table(analysis: ScopeAnalysis):
    print(f"\n  --- Memory Usage by Kernel ---")
    print(f"  {'Kernel':<28} {'Space':<8} {'Used':>10} {'Limit':>10} {'Usage':>8}  {'Bar'}")
    print(f"  {'-' * 80}")

    for entry in analysis.memory_entries:
        print(f"  {entry.kernel_name:<28} {entry.space:<8} "
              f"{entry.used_kb:>8.1f}KB {entry.limit_kb:>8.1f}KB "
              f"{_fmt_pct(entry.usage_pct):>8}  {_bar(entry.usage_pct, 15)}")


def _print_diagnostics(analysis: ScopeAnalysis):
    """Print automatic diagnostic messages."""
    issues = []

    # Check Exec/Latency ratio
    for fid, ts in analysis.task_stats.items():
        if ts.exec_latency_ratio < 50 and ts.count > 1:
            issues.append(
                f"[WARN] func {ts.func_name} (id={fid}): Exec/Latency = {_fmt_pct(ts.exec_latency_ratio)} "
                f"(< 50%) — scheduling overhead may be significant"
            )

    # Check scheduler idle
    for ss in analysis.scheduler_stats:
        if ss.idle_ratio > 30:
            issues.append(
                f"[WARN] Scheduler thread {ss.thread_id}: Idle = {_fmt_pct(ss.idle_ratio)} "
                f"(> 30%) — task submission may be insufficient"
            )

    # Check orchestrator alloc
    if analysis.orchestrator_stats:
        os = analysis.orchestrator_stats
        alloc_ratio = os.phase_ratio("orch_alloc")
        if alloc_ratio > 20:
            issues.append(
                f"[WARN] Orchestrator orch_alloc = {_fmt_pct(alloc_ratio)} "
                f"(> 20%) — ring buffer may be under pressure"
            )
        fanin_ratio = os.phase_ratio("orch_fanin")
        if fanin_ratio > 40:
            issues.append(
                f"[INFO] Orchestrator orch_fanin = {_fmt_pct(fanin_ratio)} "
                f"(> 40%) — orchestrator is waiting for predecessor tasks"
            )

    # Check memory utilization
    for entry in analysis.memory_entries:
        if entry.usage_pct < 5 and entry.space != "Acc":
            issues.append(
                f"[INFO] {entry.kernel_name}/{entry.space}: usage = {_fmt_pct(entry.usage_pct)} "
                f"(< 5%) — consider increasing tile size"
            )
        if entry.usage_pct >= 100:
            issues.append(
                f"[WARN] {entry.kernel_name}/{entry.space}: usage = 100% — at hardware limit"
            )

    if issues:
        print(f"\n  --- Diagnostics ---")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n  --- Diagnostics ---")
        print(f"  [OK] No issues detected")


def print_cross_scope_comparison(analyses: list):
    """Print a comparison table across multiple scopes."""
    if len(analyses) < 2:
        return

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  CROSS-SCOPE COMPARISON")
    print(sep)

    # Summary table
    print(f"\n  {'Scope':<40} {'Tasks':>6} {'Wall Time':>12} {'AIC Tasks':>10} {'AIV Tasks':>10} {'Exec/Lat%':>10}")
    print(f"  {'-' * 92}")

    for a in analyses:
        aic_count = sum(ts.count for ts in a.task_stats.values() if ts.core_type == "aic")
        aiv_count = sum(ts.count for ts in a.task_stats.values() if ts.core_type == "aiv")
        total_exec = sum(ts.total_exec_us for ts in a.task_stats.values())
        total_lat = sum(ts.total_latency_us for ts in a.task_stats.values())
        ratio = (total_exec / total_lat * 100) if total_lat > 0 else 0

        print(f"  {a.name:<40} {a.task_count:>6} {_fmt_us(a.wall_time_us):>12} "
              f"{aic_count:>10} {aiv_count:>10} {_fmt_pct(ratio):>10}")

    # Scheduler comparison
    print(f"\n  {'Scope':<40} {'Sched Threads':>14} {'Avg Idle%':>10}")
    print(f"  {'-' * 68}")
    for a in analyses:
        if a.scheduler_stats:
            avg_idle = sum(s.idle_ratio for s in a.scheduler_stats) / len(a.scheduler_stats)
            print(f"  {a.name:<40} {len(a.scheduler_stats):>14} {_fmt_pct(avg_idle):>10}")

    # Orchestrator comparison
    print(f"\n  {'Scope':<40} {'Orch Duration':>14} {'Top Phase':>20} {'Top%':>8}")
    print(f"  {'-' * 86}")
    for a in analyses:
        if a.orchestrator_stats:
            os = a.orchestrator_stats
            if os.phase_durations_us:
                top_phase = max(os.phase_durations_us.items(), key=lambda x: x[1])
                display = top_phase[0].replace("orch_", "")
                ratio = os.phase_ratio(top_phase[0])
                print(f"  {a.name:<40} {_fmt_us(os.total_duration_us):>14} {display:>20} {_fmt_pct(ratio):>8}")


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(analyses: list, output_path: Path):
    """Generate a structured Markdown report."""
    lines = []

    lines.append("# Performance Analysis Report\n")
    lines.append(f"Generated from `{analyses[0].path.parent if analyses else 'build_output'}/`\n")
    lines.append(f"Scopes analyzed: {len(analyses)}\n")
    lines.append("---\n")

    # Table of contents
    lines.append("## Table of Contents\n")
    for i, a in enumerate(analyses, 1):
        lines.append(f"{i}. [{a.name}](#{a.name.lower().replace(' ', '-')})")
    if len(analyses) > 1:
        lines.append(f"{len(analyses) + 1}. [Cross-Scope Comparison](#cross-scope-comparison)")
    lines.append("")

    # Per-scope sections
    for a in analyses:
        lines.extend(_md_scope_section(a))

    # Cross-scope comparison
    if len(analyses) > 1:
        lines.extend(_md_cross_scope(analyses))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _md_scope_section(a: ScopeAnalysis) -> list:
    lines = []
    lines.append(f"\n## {a.name}\n")
    lines.append(f"- **Path**: `{a.path}`")
    lines.append(f"- **Runtime**: {a.runtime}")
    lines.append(f"- **Block Dim**: {a.block_dim}")
    lines.append(f"- **AICPU Threads**: {a.aicpu_thread_num}")
    lines.append(f"- **Task Count**: {a.task_count}")
    lines.append(f"- **Wall Time**: {_fmt_us(a.wall_time_us)}")
    lines.append("")

    # Task table
    if a.task_stats:
        lines.append("### Task Statistics\n")
        lines.append("| FuncID | Name | Type | Count | Avg Exec | Avg Latency | Exec% | Avg HeadOH | Avg TailOH |")
        lines.append("|--------|------|------|------:|----------|-------------|------:|------------|------------|")
        total_exec = 0.0
        total_lat = 0.0
        total_count = 0
        for fid in sorted(a.task_stats.keys()):
            ts = a.task_stats[fid]
            total_exec += ts.total_exec_us
            total_lat += ts.total_latency_us
            total_count += ts.count
            lines.append(
                f"| {fid} | {ts.func_name} | {ts.core_type} | {ts.count} | "
                f"{_fmt_us(ts.avg_exec_us)} | {_fmt_us(ts.avg_latency_us)} | "
                f"{_fmt_pct(ts.exec_latency_ratio)} | {_fmt_us(ts.avg_head_oh_us)} | "
                f"{_fmt_us(ts.avg_tail_oh_us)} |"
            )
        overall_ratio = (total_exec / total_lat * 100) if total_lat > 0 else 0
        lines.append(
            f"| **TOTAL** | | | **{total_count}** | "
            f"**{_fmt_us(total_exec)}** | **{_fmt_us(total_lat)}** | "
            f"**{_fmt_pct(overall_ratio)}** | | |"
        )
        lines.append("")

    # Scheduler table
    if a.scheduler_stats:
        lines.append("### Scheduler Statistics\n")
        all_phases = set()
        for ss in a.scheduler_stats:
            all_phases.update(ss.phase_durations_us.keys())
        phase_order = ["complete", "dispatch", "scan", "idle"]
        phases = [p for p in phase_order if p in all_phases]

        header = "| Thread | Total |"
        divider = "|--------|-------|"
        for p in phases:
            header += f" {p} |"
            divider += "------|"
        header += " Idle% |"
        divider += "------|"
        lines.append(header)
        lines.append(divider)

        for ss in a.scheduler_stats:
            row = f"| {ss.thread_id} | {_fmt_us(ss.total_duration_us)} |"
            for p in phases:
                row += f" {_fmt_us(ss.phase_durations_us.get(p, 0))} |"
            row += f" {_fmt_pct(ss.idle_ratio)} |"
            lines.append(row)
        lines.append("")

    # Orchestrator table
    if a.orchestrator_stats:
        os = a.orchestrator_stats
        lines.append("### Orchestrator Statistics\n")
        lines.append(f"- **Total Duration**: {_fmt_us(os.total_duration_us)}")
        lines.append(f"- **Tasks Submitted**: {os.total_tasks}")
        lines.append("")

        phase_order = ["orch_alloc", "orch_sync", "orch_lookup", "orch_insert",
                        "orch_params", "orch_fanin", "orch_scope_end"]
        all_phases = sorted(os.phase_durations_us.keys())
        phases = [p for p in phase_order if p in os.phase_durations_us]
        phases += [p for p in all_phases if p not in phase_order]

        lines.append("| Phase | Duration | Count | Ratio |")
        lines.append("|-------|----------|------:|------:|")
        for phase in phases:
            dur = os.phase_durations_us[phase]
            cnt = os.phase_counts[phase]
            ratio = os.phase_ratio(phase)
            display = phase.replace("orch_", "") if phase.startswith("orch_") else phase
            lines.append(f"| {display} | {_fmt_us(dur)} | {cnt} | {_fmt_pct(ratio)} |")
        lines.append("")

    # Memory table
    if a.memory_entries:
        lines.append("### Memory Usage\n")
        lines.append("| Kernel | Space | Used | Limit | Usage |")
        lines.append("|--------|-------|-----:|------:|------:|")
        for entry in a.memory_entries:
            lines.append(
                f"| {entry.kernel_name} | {entry.space} | "
                f"{entry.used_kb:.1f} KB | {entry.limit_kb:.1f} KB | {_fmt_pct(entry.usage_pct)} |"
            )
        lines.append("")

    # Diagnostics
    diagnostics = _collect_diagnostics(a)
    if diagnostics:
        lines.append("### Diagnostics\n")
        for d in diagnostics:
            lines.append(f"- {d}")
        lines.append("")

    return lines


def _collect_diagnostics(a: ScopeAnalysis) -> list:
    issues = []
    for fid, ts in a.task_stats.items():
        if ts.exec_latency_ratio < 50 and ts.count > 1:
            issues.append(
                f"**WARN** `{ts.func_name}` (func_id={fid}): Exec/Latency = {_fmt_pct(ts.exec_latency_ratio)} "
                f"(< 50%) — scheduling overhead may be significant"
            )
    for ss in a.scheduler_stats:
        if ss.idle_ratio > 30:
            issues.append(
                f"**WARN** Scheduler thread {ss.thread_id}: Idle = {_fmt_pct(ss.idle_ratio)} "
                f"(> 30%) — task submission may be insufficient"
            )
    if a.orchestrator_stats:
        os = a.orchestrator_stats
        alloc_ratio = os.phase_ratio("orch_alloc")
        if alloc_ratio > 20:
            issues.append(
                f"**WARN** Orchestrator orch_alloc = {_fmt_pct(alloc_ratio)} "
                f"(> 20%) — ring buffer may be under pressure"
            )
        fanin_ratio = os.phase_ratio("orch_fanin")
        if fanin_ratio > 40:
            issues.append(
                f"**INFO** Orchestrator orch_fanin = {_fmt_pct(fanin_ratio)} "
                f"(> 40%) — orchestrator is waiting for predecessor tasks"
            )
    for entry in a.memory_entries:
        if entry.usage_pct < 5 and entry.space != "Acc":
            issues.append(
                f"**INFO** `{entry.kernel_name}`/{entry.space}: usage = {_fmt_pct(entry.usage_pct)} "
                f"(< 5%) — consider increasing tile size"
            )
        if entry.usage_pct >= 100:
            issues.append(
                f"**WARN** `{entry.kernel_name}`/{entry.space}: usage = 100% — at hardware limit"
            )
    return issues


def _md_cross_scope(analyses: list) -> list:
    lines = []
    lines.append("\n## Cross-Scope Comparison\n")

    lines.append("### Overview\n")
    lines.append("| Scope | Tasks | Wall Time | AIC Tasks | AIV Tasks | Exec/Latency |")
    lines.append("|-------|------:|-----------|----------:|----------:|-------------:|")
    for a in analyses:
        aic = sum(ts.count for ts in a.task_stats.values() if ts.core_type == "aic")
        aiv = sum(ts.count for ts in a.task_stats.values() if ts.core_type == "aiv")
        te = sum(ts.total_exec_us for ts in a.task_stats.values())
        tl = sum(ts.total_latency_us for ts in a.task_stats.values())
        ratio = (te / tl * 100) if tl > 0 else 0
        lines.append(f"| {a.name} | {a.task_count} | {_fmt_us(a.wall_time_us)} | {aic} | {aiv} | {_fmt_pct(ratio)} |")
    lines.append("")

    # Scheduler comparison
    lines.append("### Scheduler Comparison\n")
    lines.append("| Scope | Active Threads | Avg Idle% |")
    lines.append("|-------|---------------:|----------:|")
    for a in analyses:
        if a.scheduler_stats:
            avg_idle = sum(s.idle_ratio for s in a.scheduler_stats) / len(a.scheduler_stats)
            lines.append(f"| {a.name} | {len(a.scheduler_stats)} | {_fmt_pct(avg_idle)} |")
    lines.append("")

    # Orchestrator comparison
    lines.append("### Orchestrator Comparison\n")
    lines.append("| Scope | Duration | Top Phase | Top% |")
    lines.append("|-------|----------|-----------|-----:|")
    for a in analyses:
        if a.orchestrator_stats:
            os = a.orchestrator_stats
            if os.phase_durations_us:
                top = max(os.phase_durations_us.items(), key=lambda x: x[1])
                display = top[0].replace("orch_", "")
                ratio = os.phase_ratio(top[0])
                lines.append(f"| {a.name} | {_fmt_us(os.total_duration_us)} | {display} | {_fmt_pct(ratio)} |")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze performance data in build_output/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s build_output/                              # Analyze all scopes
              %(prog)s build_output/Qwen3Scope1_20260419_204240/  # Single scope
              %(prog)s build_output/ -o my_report.md              # Custom output
              %(prog)s build_output/ -v                           # Verbose
        """),
    )
    parser.add_argument("path", help="Path to build_output/ directory or a single scope directory")
    parser.add_argument("-o", "--output", help="Output Markdown report path (default: <path>/perf_report.md)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser


def main():
    args = _build_parser().parse_args()
    base_path = Path(args.path)

    if not base_path.exists():
        print(f"Error: Path not found: {base_path}", file=sys.stderr)
        return 1

    # Discover scopes
    scopes = discover_scopes(base_path)
    if not scopes:
        print(f"Error: No scope directories found in {base_path}", file=sys.stderr)
        print("A scope directory must contain kernel_config.py", file=sys.stderr)
        return 1

    print(f"Found {len(scopes)} scope(s) to analyze")

    # Analyze each scope
    analyses = []
    for scope_dir in scopes:
        if args.verbose:
            print(f"\nAnalyzing: {scope_dir.name}")
        analysis = analyze_scope(scope_dir, verbose=args.verbose)
        if analysis:
            analyses.append(analysis)

    if not analyses:
        print("Error: No valid analyses produced", file=sys.stderr)
        return 1

    # Terminal output
    for a in analyses:
        print_scope_summary(a)

    if len(analyses) > 1:
        print_cross_scope_comparison(analyses)

    # Markdown report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base_path / "perf_report.md"

    generate_markdown_report(analyses, output_path)
    print(f"\n{'=' * 100}")
    print(f"  Markdown report written to: {output_path}")
    print(f"{'=' * 100}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
