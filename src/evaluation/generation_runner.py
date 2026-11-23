"""Run generation, feature extraction, and MAUVE for one trained checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .generation_protocol import (
    REPO,
    SEEDS,
    MODES,
    SAMPLE_COUNT,
    SCORER_PROTOCOL,
    atomic_json,
    check_versions,
    file_sha,
    json_sha,
    load_records,
    runtime_env,
    source_fingerprints,
)


def now():
    return dt.datetime.now().astimezone().isoformat()


def gpu_inventory():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in raw.splitlines():
        index, uuid, name, driver, memory, utilization = [
            s.strip() for s in line.split(",")
        ]
        rows.append(
            dict(
                index=int(index),
                uuid=uuid,
                name=name,
                driver=driver,
                memory_mib=int(memory),
                utilization=int(utilization),
            )
        )
    return rows


def select_gpus(inventory, requested, workers):
    if not 1 <= workers <= SAMPLE_COUNT:
        raise ValueError("Invalid number of GPU workers")
    choices = (
        requested.split(",")
        if requested is not None
        else [str(r["index"]) for r in inventory[:workers]]
    )
    if len(choices) != workers or any(not c.strip() for c in choices):
        raise ValueError("The GPU selection must contain exactly --workers devices")
    selected = []
    for choice in choices:
        choice = choice.strip()
        matches = [
            r for r in inventory if choice == str(r["index"]) or r["uuid"] == choice
        ]
        if len(matches) != 1:
            raise ValueError(f"GPU not found or ambiguous: {choice}")
        selected.append(matches[0])
    if len({r["uuid"] for r in selected}) != workers:
        raise ValueError("Duplicate GPU selection")
    return selected


def gpu_processes(uuids):
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        dict(pid=int(fields[0]), gpu_uuid=fields[1].strip(), memory=fields[2].strip())
        for line in raw.splitlines()
        if line.strip()
        for fields in [line.split(",")]
        if fields[1].strip() in uuids
    ]


def is_owned(pid, roots):
    """Include a worker's descendants without allowing unrelated GPU processes."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid in roots:
            return True
        seen.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            return False
        pid = int(fields[1])
    return False


def stop_children(children):
    # Only process groups created by this runner are signalled.
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    while (
        any(child.poll() is None for child in children) and time.monotonic() < deadline
    ):
        time.sleep(0.2)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait()


def verify_experiment(experiment):
    if json_sha(experiment["protocol"]) != experiment["fingerprint"]:
        raise ValueError("Experiment metadata fingerprint mismatch")
    if source_fingerprints() != experiment["protocol"]["source_sha256"]:
        raise RuntimeError(
            "Evaluation source changed; refusing mixed-implementation results"
        )


def build_experiment(args, selected):
    from .generation_data import verify_prepared
    from src.inference.generation_kernels import build_library, compiler_config

    inputs = verify_prepared(args.inputs)
    versions = check_versions()
    native = {
        name: file_sha(build_library(name))
        for name in ("exit_control", "depth_scheduler", "rms_norm", "rotary")
    }
    protocol = dict(
        label=args.label,
        recurrence=args.recurrence,
        checkpoint_step=args.expected_step,
        checkpoint_sha256=file_sha(args.checkpoint),
        manifest_sha256=inputs["manifest_sha256"],
        tokenizer_files=inputs["tokenizer_files"],
        scorer_files=inputs["scorer_files"],
        scorer=SCORER_PROTOCOL,
        modes=list(MODES),
        seeds=list(SEEDS),
        samples_per_job=SAMPLE_COUNT,
        prompt_tokens=64,
        max_new_tokens=512,
        temperature=1.0,
        top_k=None,
        top_p=None,
        stop_on_eos=True,
        batch_size=1,
        world_size=args.workers,
        feature_batch_size=8,
        exit_head_use_final_norm=False,
        grt_training_noise_enabled=False,
        precision="fp32_parameters_bf16_autocast_shared_bf16_loop_linears",
        rmsnorm="full_fused",
        rope="fused",
        scheduler="owned_depth_streams",
        compute_connections=32,
        copy_connections=32,
        versions=versions,
        source_sha256=source_fingerprints(),
        native_sha256=native,
        nvcc=compiler_config()[2],
        host=socket.gethostname(),
        gpus=[
            {key: gpu[key] for key in ("uuid", "name", "driver")} for gpu in selected
        ],
    )
    return dict(
        protocol=protocol,
        fingerprint=json_sha(protocol),
        created=now(),
        paths=dict(
            checkpoint=str(Path(args.checkpoint).resolve()),
            inputs=str(Path(args.inputs).resolve()),
            manifest=str(Path(args.inputs).resolve() / "prompts.jsonl"),
            tokenizer=inputs["tokenizer_path"],
            scorer=inputs["scorer_path"],
        ),
    )


def job_plan(workers):
    return (
        [
            dict(name="generation_check", action="check-generation", workers=1),
            dict(name="scorer_check", action="check-scorer", workers=1),
        ]
        + [
            dict(
                name=f"mode{mode}_seed{seed}",
                action="generate",
                workers=workers,
                mode=mode,
                seed=seed,
            )
            for seed in SEEDS
            for mode in MODES
        ]
        + [
            dict(name="features", action="features", workers=workers),
            dict(name="mauve", action="score", workers=1),
        ]
    )


def run(args):
    checkpoint, inputs, output = (
        Path(args.checkpoint),
        Path(args.inputs),
        Path(args.output),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not (inputs / "prepared.json").is_file():
        raise FileNotFoundError(f"Run scripts/prepare_generation.py first: {inputs}")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    label=args.label,
                    recurrence=args.recurrence,
                    checkpoint=str(checkpoint.resolve()),
                    inputs=str(inputs.resolve()),
                    output=str(output.resolve()),
                    generations=45000,
                    plan=job_plan(args.workers),
                ),
                indent=2,
            )
        )
        return
    selected = select_gpus(gpu_inventory(), args.gpus, args.workers)
    uuids = {gpu["uuid"] for gpu in selected}
    if gpu_processes(uuids) or any(
        gpu["utilization"] > 5 or gpu["memory_mib"] > 512 for gpu in selected
    ):
        raise RuntimeError(
            "Selected GPUs are occupied. Obtain an exclusive allocation before starting timing measurements."
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "timing_invalidated.json").exists():
            raise RuntimeError(
                "This output contains a GPU-contaminated timing job. Preserve it for inspection and choose a fresh output directory."
            )
        experiment = build_experiment(args, selected)
        experiment_path = output / "experiment.json"
        if experiment_path.exists():
            previous = json.loads(experiment_path.read_text())
            if previous["fingerprint"] != experiment["fingerprint"]:
                raise ValueError(
                    "Output directory belongs to a different checkpoint, hardware, or evaluation protocol"
                )
        atomic_json(experiment_path, experiment)
        (output / "logs").mkdir(exist_ok=True)
        state = dict(
            status="running",
            started=now(),
            current_job=None,
            completed_jobs=[],
            generation_records=0,
        )
        children = []
        original_handlers = {}

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            original_handlers[sig] = signal.signal(sig, interrupted)
        try:
            for job in job_plan(args.workers):
                verify_experiment(experiment)
                if job["action"] in ("features", "score"):
                    state["generation_records"] = sum(
                        len(load_records(output, experiment, mode, seed))
                        for seed in SEEDS
                        for mode in MODES
                    )
                    if state["generation_records"] != 45000:
                        raise ValueError(
                            "Exactly 45,000 complete generations are required before scoring"
                        )
                deadline = time.monotonic() + 30
                while gpu_processes(uuids) and time.monotonic() < deadline:
                    time.sleep(1)
                if gpu_processes(uuids):
                    raise RuntimeError(
                        "Another process occupies the selected GPUs; stopping before the next stage"
                    )
                state["current_job"] = job["name"]
                atomic_json(output / "state.json", state)
                print(f"JOB_STARTED {job['name']} {now()}", flush=True)
                started = time.monotonic()
                handles, children = [], []
                try:
                    for rank in range(job["workers"]):
                        command = [
                            sys.executable,
                            "-u",
                            str(REPO / "scripts/evaluate_generation.py"),
                            "--action",
                            job["action"],
                            "--output",
                            str(output.resolve()),
                            "--rank",
                            str(rank),
                        ]
                        if job["action"] == "generate":
                            command += [
                                "--mode",
                                str(job["mode"]),
                                "--seed",
                                str(job["seed"]),
                            ]
                        log = (output / "logs" / f"{job['name']}_rank{rank}.out").open(
                            "a", buffering=1
                        )
                        handles.append(log)
                        env = {
                            **runtime_env(),
                            "CUDA_VISIBLE_DEVICES": selected[rank]["uuid"],
                        }
                        if job["action"] == "score":
                            env.update(
                                CUDA_VISIBLE_DEVICES="",
                                OMP_NUM_THREADS="8",
                                MKL_NUM_THREADS="8",
                            )
                        children.append(
                            subprocess.Popen(
                                command,
                                cwd=REPO,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        )
                    roots = {child.pid for child in children}
                    while True:
                        apps = gpu_processes(uuids)
                        foreign = [
                            app for app in apps if not is_owned(app["pid"], roots)
                        ]
                        with (output / "resource_monitor.jsonl").open("a") as monitor:
                            monitor.write(
                                json.dumps(dict(time=now(), job=job["name"], apps=apps))
                                + "\n"
                            )
                        if foreign:
                            if job["action"] == "generate":
                                atomic_json(
                                    output / "timing_invalidated.json",
                                    dict(job=job["name"], time=now(), foreign=foreign),
                                )
                            raise RuntimeError(
                                "Foreign GPU activity detected; stopped this evaluation's workers"
                            )
                        failures = [
                            (c.pid, c.returncode)
                            for c in children
                            if c.poll() not in (None, 0)
                        ]
                        if failures:
                            raise RuntimeError(
                                f"Worker failed in {job['name']}: {failures}; see logs/"
                            )
                        if all(child.poll() is not None for child in children):
                            break
                        time.sleep(10)
                finally:
                    stop_children(children)
                    for handle in handles:
                        handle.close()
                state["completed_jobs"].append(
                    dict(job=job["name"], wall_seconds=time.monotonic() - started)
                )
                atomic_json(output / "state.json", state)
                print(f"JOB_COMPLETE {job['name']} {now()}", flush=True)
            verify_experiment(experiment)
            summary = json.loads((output / "summary.json").read_text())
            if len(summary["per_seed"]) != 9 or any(
                r["n"] != 5000 for r in summary["per_seed"]
            ):
                raise ValueError("Incomplete quality results")
            state.update(status="complete", finished=now(), current_job=None)
            atomic_json(output / "state.json", state)
            print(f"EXPERIMENT_COMPLETE {output / 'RESULTS.md'}", flush=True)
        except BaseException as exc:
            stop_children(children)
            state.update(
                status=(
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                ),
                error=repr(exc),
                stopped=now(),
            )
            atomic_json(output / "state.json", state)
            raise
        finally:
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "run",
            "generate",
            "check-generation",
            "check-scorer",
            "features",
            "score",
        ),
        default="run",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--inputs", default="data/generation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recurrence", choices=("vanilla", "grt"))
    parser.add_argument("--label", default="Trained looped language model")
    parser.add_argument("--expected-step", type=int, default=22000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--mode", type=int, choices=MODES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "run":
        if not args.checkpoint or not args.recurrence:
            parser.error("--checkpoint and --recurrence are required")
        run(args)
        return
    experiment = json.loads((Path(args.output) / "experiment.json").read_text())
    verify_experiment(experiment)
    if args.action in ("generate", "check-generation"):
        from .generation_worker import check_generation, generation_job

        if args.action == "generate":
            if args.mode is None or args.seed is None:
                parser.error("Generation requires --mode and --seed")
            generation_job(experiment, args.output, args.mode, args.seed, args.rank)
        else:
            check_generation(experiment, args.output)
    else:
        from .generation_scoring import check_scorer, feature_job, aggregate_job

        if args.action == "features":
            feature_job(experiment, args.output, args.rank)
        elif args.action == "score":
            aggregate_job(experiment, args.output)
        else:
            check_scorer(experiment, args.output)
