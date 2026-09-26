from __future__ import annotations

import argparse
from pathlib import Path

from ucfm.config import load_config
from ucfm.metrics import update_readme
from ucfm.pipeline import ensure_eval_present, run_baseline_phase, train_and_eval
from ucfm.registry import load_registry
from ucfm.s3ops import check_credentials, s3_prefix_exists, sync_prefix_from_s3, sync_prefix_to_s3
from ucfm.storage import Store


def _store(args) -> Store:
    cfg = load_config(Path(args.config) if getattr(args, "config", None) else None)
    return Store(cfg)


def cmd_baseline(args) -> None:
    store = _store(args)
    print(run_baseline_phase(store, force=args.force))


def cmd_train(args) -> None:
    store = _store(args)
    print(train_and_eval(store))


def cmd_sync_eval(args) -> None:
    store = _store(args)
    n = sync_prefix_from_s3(store, "eval")
    print(f"synced {n} files into data/eval/")


def cmd_sync_pool(args) -> None:
    store = _store(args)
    n = sync_prefix_from_s3(store, "pool")
    print(f"synced {n} files into data/pool/")


def cmd_upload_eval(args) -> None:
    store = _store(args)
    ensure_eval_present(store)
    n = sync_prefix_to_s3(store, "eval")
    print(f"uploaded {n} files to s3://{store.cfg.storage.bucket}/eval/")


def cmd_s3_check(args) -> None:
    store = _store(args)
    print(check_credentials())
    print(f"bucket={store.cfg.storage.bucket} backend={store.cfg.storage.backend}")
    for prefix in ("eval", "pool", "logs"):
        print(f"  {prefix}/ exists on s3: {s3_prefix_exists(store, prefix)}")


def cmd_registry(args) -> None:
    for row in load_registry():
        print(row)


def cmd_update_readme(args) -> None:
    store = _store(args)
    print(update_readme(store))


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default=None)
    p = argparse.ArgumentParser(
        prog="ucfm",
        description="UAV crop foundation model — manual S3 ingest + DINOv2 train/eval",
        parents=[parent],
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("baseline", help="DINOv2 probe on ws/gs (no pool pretrain)", parents=[parent])
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("train", help="Pretrain on pool + eval ws/gs", parents=[parent])
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("sync-eval", help="Download eval/ from S3 to data/eval/", parents=[parent])
    s.set_defaults(func=cmd_sync_eval)

    s = sub.add_parser("sync-pool", help="Download pool/ from S3 to data/pool/", parents=[parent])
    s.set_defaults(func=cmd_sync_pool)

    s = sub.add_parser("upload-eval", help="Upload data/eval/ to S3", parents=[parent])
    s.set_defaults(func=cmd_upload_eval)

    s = sub.add_parser("s3-check", help="Verify AWS credentials and bucket prefixes", parents=[parent])
    s.set_defaults(func=cmd_s3_check)

    s = sub.add_parser("registry", help="Print datasets/registry.yaml", parents=[parent])
    s.set_defaults(func=cmd_registry)

    s = sub.add_parser(
        "update-readme",
        help="Regenerate assets/performance.png + README results section",
        parents=[parent],
    )
    s.set_defaults(func=cmd_update_readme)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
