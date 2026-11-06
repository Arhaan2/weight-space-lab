# Execution diagnostics

Run the environment inventory without loading a model or optional inference libraries:

```bash
python -m weight_space_lab diagnose
```

The default report uses Python's standard library and installed package metadata. It records CPU architecture and logical count, host RAM, dependency versions, and explicit unprobed backend states. Package installation is not proof that a library can import or execute. Missing packages and unavailable measurements remain explicit; no guessed version or zero-memory success is substituted.

To inspect the installed PyTorch runtime and its visible CPU, CUDA, and MPS devices:

```bash
python -m weight_space_lab diagnose --probe-backends
```

This opt-in probe runs in an isolated child process with a ten-second timeout. It imports the locally installed runtime and queries availability and memory; it does not load weights, allocate model tensors, download artifacts, generate outputs, or select a device for a later run. A timeout or import failure is a diagnostic outcome, not model validation. CI tests use mocked backend responses and never require an accelerator.

CUDA free/total memory uses the runtime's [memory query](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.mem_get_info.html). MPS reports its recommended working-set maximum and current allocation where those [runtime APIs](https://docs.pytorch.org/docs/stable/mps.html) are available. MPS's recommendation is not free memory. Host memory is not a container or process memory limit, and a memory snapshot cannot establish checkpoint fit. Linux available RAM uses the kernel estimate; macOS uses explicitly labeled free, inactive, and speculative pages. Unsupported observations are null.

The report excludes usernames, hostnames, environment variables, package locations, executable paths, provider device names, and raw exception or subprocess text. Device labels are backend identifiers such as `cuda:0`. Version strings use a restricted public-version grammar; unusual local builds are marked `unreportable` instead of exposing arbitrary metadata. Error states use fixed labels. No credentials are inspected or reported.

Backend discovery cannot verify the study's architecture, precision, tokenizer, decoding behavior, throughput, frozen protocol, or editor connection. Before real-model validation, install the selected compatible runtime deliberately, resolve immutable assets and settings, then perform a bounded development-only execution under the appropriate profile. Browser readiness requires a separate actual browser session and remains `not_checked` here.
