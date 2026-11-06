"""Operator commands for offline Phase 7 integration (never implicit live calls)."""

from pathlib import Path


def add_commands(commands):
    init = commands.add_parser("editor-init", help="freeze an offline development editor session and durable budget")
    for name in ("root", "model", "policy", "development", "limits", "interface-observation"):
        init.add_argument("--" + name, type=Path, required=True)
    init.add_argument("--probe", type=Path, help="optional frozen development-only forward-pass specification")
    for name, help_text in (
        ("editor-status", "show measured, reserved and conservatively charged budget"),
        ("editor-cancel", "persist cancellation and stop active dispatch through its execution monitor"),
        ("editor-recover", "reconcile interrupted requests without refund or replay"),
        ("editor-export", "reserve and write a provider-independent offline request"),
        ("editor-import", "consume an unverified offline file response; never authenticate model identity"),
        ("editor-simulate", "exercise the OS-isolated fixture; no model is called"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--root", type=Path, required=True)
        if name == "editor-export":
            parser.add_argument("--instruction", required=True)
            parser.add_argument("--max-output-tokens", type=int, default=512)
            parser.add_argument("--timeout-seconds", type=int, default=60)
            parser.add_argument("--retry-of")
            parser.add_argument("--feedback-id")
        elif name == "editor-import":
            parser.add_argument("--response", type=Path, required=True)
        elif name == "editor-simulate":
            parser.add_argument("--tool", choices=["model_info", "development", "inspect", "propose", "probe", "stop"], default="model_info")
            parser.add_argument("--arguments", type=Path)
    check = commands.add_parser("editor-check-isolation", help="run hostile filesystem/network denial checks in the actual OS sandbox")
    check.add_argument("--out", type=Path)
    readiness = commands.add_parser("editor-live-readiness", help="inspect exact live dispatch blockers without making a model request")
    readiness.add_argument("--root", type=Path, help="read an existing session budget without resuming or changing it")
    readiness.add_argument("--out", type=Path)
    connection = commands.add_parser("editor-prepare-connection", help="save the inert zero-action development connection plan; never dispatch")
    connection.add_argument("--out", type=Path, required=True)


def run(args):
    from .editor_contract import read_json_bounded
    from .editor_controller import EditorController
    from .native_runner import _write_json
    if args.command == "editor-live-readiness":
        from .editor_live import live_readiness
        result = live_readiness(root=args.root)
        if args.out is not None:
            _write_json(args.out, result, immutable=True)
        return result
    if args.command == "editor-prepare-connection":
        from .editor_live import connection_test_plan
        result = connection_test_plan()
        _write_json(args.out, result, immutable=True)
        return result
    if args.command == "editor-check-isolation":
        from .editor_sandbox import verify_denials
        result = verify_denials()
        if args.out is not None:
            _write_json(args.out, result, immutable=True)
        return result
    if args.command == "editor-init":
        controller = EditorController.create(args.root, model=args.model, policy=read_json_bounded(args.policy),
            development=read_json_bounded(args.development), limits=read_json_bounded(args.limits),
            interface_observation=read_json_bounded(args.interface_observation),
            probe=None if args.probe is None else read_json_bounded(args.probe))
        try:
            return controller.status()
        finally:
            controller.ledger.close()
    controller = EditorController(args.root)
    try:
        return _session_command(args, controller, read_json_bounded)
    finally:
        controller.ledger.close()


def _session_command(args, controller, read_json_bounded):
    if args.command == "editor-status":
        return controller.status()
    if args.command == "editor-cancel":
        return controller.cancel()
    if args.command == "editor-recover":
        return controller.recover()
    if args.command == "editor-export":
        request = controller.export_request(args.instruction, max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds, retry_of=args.retry_of, feedback_id=args.feedback_id)
        return {"request_id": request["request_id"], "handoff_file": f"requests/{request['request_id']}.json",
                "mode": "offline_integration", "scientific_result": False, "experiment_executed": False}
    if args.command == "editor-import":
        return controller.import_response(read_json_bounded(args.response))
    if args.command == "editor-simulate":
        return controller.simulate(tool=args.tool,
            arguments=None if args.arguments is None else read_json_bounded(args.arguments))
    raise ValueError("unknown editor command")
