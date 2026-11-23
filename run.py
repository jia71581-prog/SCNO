"""List, inspect, verify, and run SCNO training and fine-tuning workflows."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings


ROOT = Path(__file__).resolve().parent
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))


def load_json(name):
    return json.loads((ROOT / "docs" / name).read_text(encoding="utf-8"))


def show_workflow(name, workflow):
    print(f"{name} [{workflow['group']}]\n  Entry point: {workflow['script']}")
    for key, value in workflow.get("path_environment", {}).items():
        print(f"  {key}={value}")
    print(f"  Arguments: python run.py run {name} -- --help")


def run_workflow(workflow, interpreter, arguments):
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    environment = os.environ.copy()
    paths = [str(ROOT / p) for p in workflow["pythonpath"]] + [str(ROOT)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["ACOUSTIC_RESEARCH_ROOT"] = str(ROOT)
    for key, path in workflow.get("path_environment", {}).items():
        environment[key] = str(ROOT / path)
    program = ["-m", workflow["module"]] if "module" in workflow else [str(ROOT / workflow["script"])]
    return subprocess.run([interpreter, *program, *arguments], env=environment).returncode


def check(workflows):
    files = load_json("source_hashes.json")
    errors = []
    for name, expected in files.items():
        try:
            payload = (ROOT / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                errors.append(f"checksum mismatch: {name}")
            if name.endswith(".py"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    compile(payload, name, "exec")
        except (OSError, SyntaxError, ValueError) as error:
            errors.append(f"{name}: {error}")
    for name, item in workflows.items():
        if not (ROOT / item["script"]).is_file():
            errors.append(f"missing entry point: {name}")
        for path in [*item["pythonpath"], *item.get("path_environment", {}).values()]:
            if not (ROOT / path).is_dir():
                errors.append(f"missing import path: {name} {path}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {len(files)} source and configuration files, {len(workflows)} workflows, {len(errors)} errors.")
    return int(bool(errors))


def main():
    workflows = load_json("workflows.json")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list workflows")
    listing.add_argument("group", nargs="?", choices=("train", "finetune"))
    describe = commands.add_parser("show", help="display the registered entry point")
    describe.add_argument("workflow", choices=workflows)
    run = commands.add_parser("run", help="run a workflow with its own CLI arguments")
    run.add_argument("--python", default=os.environ.get("ACOUSTIC_PYTHON", sys.executable))
    run.add_argument("workflow", choices=workflows)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    commands.add_parser("check", help="check source hashes, syntax and entry points without importing PyTorch")
    args = parser.parse_args()
    if args.command == "check":
        return check(workflows)
    if args.command == "run":
        return run_workflow(workflows[args.workflow], args.python, args.arguments)
    if args.command == "show":
        show_workflow(args.workflow, workflows[args.workflow])
    else:
        for name, item in workflows.items():
            if not args.group or args.group == item["group"]:
                print(f"{item['group']:9s} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
