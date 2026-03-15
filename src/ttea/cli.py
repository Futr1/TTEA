from __future__ import annotations

import argparse
import json
import sys

from .agents import TopologyFactory
from .config import load_experiment_config, load_platform_config
from .datasets import DatasetLoaderFactory, DatasetRegistry
from .exceptions import TTEAError
from .experiments import build_runner
from .types import TaskGroup


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttea", description="TTEA platform CLI")
    parser.add_argument("--platform", default="configs/platform.json", help="Path to the platform configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("describe-datasets", help="List datasets and local availability.")

    topology_parser = subparsers.add_parser("build-topology", help="Inspect the agent topology for a task group.")
    topology_parser.add_argument("--task-group", required=True, choices=[group.value for group in TaskGroup])

    plan_parser = subparsers.add_parser("plan-experiment", help="Inspect an experiment configuration.")
    plan_parser.add_argument("--experiment", required=True, help="Path to an experiment JSON file.")

    dry_run_parser = subparsers.add_parser("dry-run", help="Preview a small number of tasks.")
    dry_run_parser.add_argument("--experiment", required=True, help="Path to an experiment JSON file.")
    dry_run_parser.add_argument("--limit", type=int, default=2, help="Number of tasks to preview.")

    run_parser = subparsers.add_parser("run-experiment", help="Execute an experiment with available local data.")
    run_parser.add_argument("--experiment", required=True, help="Path to an experiment JSON file.")
    run_parser.add_argument("--limit", type=int, default=None, help="Optional task limit.")
    run_parser.add_argument("--split", default="test", help="Dataset split to use.")
    run_parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Use placeholder tasks when the dataset is not available locally.",
    )

    train_parser = subparsers.add_parser("train-experiment", help="Build the configured Transformers training pipeline.")
    train_parser.add_argument("--experiment", required=True, help="Path to an experiment JSON file.")
    train_parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Use placeholder tasks when the dataset is not available locally.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        platform_config = load_platform_config(args.platform)

        if args.command == "describe-datasets":
            registry = DatasetRegistry(platform_config.paths.data_root, platform_config.root_dir)
            payload = []
            for descriptor in registry.all():
                loader = DatasetLoaderFactory.create(descriptor)
                payload.append({**descriptor.to_dict(), "available": loader.is_available()})
            _print(payload)
            return

        if args.command == "build-topology":
            topology = TopologyFactory(platform_config).build(TaskGroup(args.task_group))
            _print(topology.describe())
            return

        experiment_config = load_experiment_config(args.experiment)
        runner = build_runner(platform_config, experiment_config)

        if args.command == "plan-experiment":
            _print(runner.plan())
            return

        if args.command == "dry-run":
            _print({"experiment": experiment_config.name, "preview_tasks": runner.preview_tasks(limit=args.limit)})
            return

        if args.command == "run-experiment":
            _print(runner.run(split=args.split, limit=args.limit, allow_placeholder=args.allow_placeholder))
            return

        if args.command == "train-experiment":
            if not experiment_config.training.enabled:
                raise TTEAError(f"Training is disabled for experiment {experiment_config.name}.")
            _print(runner.train(allow_placeholder=args.allow_placeholder))
            return
    except TTEAError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
