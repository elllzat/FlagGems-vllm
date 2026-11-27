# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Load only the pinned native operator, without importing all vLLM models."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path


def _verify_api_resolution(binding, expected_api):
    """Check the pinned adapter's real resolver, not just a file on disk."""

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        ]

    handle = ctypes.CDLL(str(binding))
    # Itanium ABI symbol exported by this pinned upstream C++ adapter.
    resolve = handle._Z16GetOpApiFuncAddrPKc
    resolve.argtypes, resolve.restype = [ctypes.c_char_p], ctypes.c_void_p
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    entries = []
    for name in ("aclnnKdaGateCumsumVllm", "aclnnKdaGateCumsumVllmGetWorkspaceSize"):
        address, info = resolve(name.encode()), DlInfo()
        if not address or not dladdr(address, ctypes.byref(info)):
            raise RuntimeError(f"Cannot identify the resolved native entry: {name}")
        actual = Path(info.filename.decode()).resolve()
        if actual != expected_api.resolve():
            raise RuntimeError(f"Native entry {name} resolves to unexpected {actual}")
        entries.append({"symbol": name, "library": str(actual)})
    return entries


def load_reference(source, install, build):
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    source, install, build = map(lambda p: Path(p).resolve(), (source, install, build))
    manifest = json.loads(Path(__file__).with_name("ascend_source.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != manifest["commit"]:
        raise RuntimeError(f"Reference checkout {revision} differs from pinned commit")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD"], check=True
    )
    for spec in manifest["sources"]:
        data = (source / spec["path"]).read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != spec["git_blob_sha"]:
            raise RuntimeError(f"Changed reference source: {spec['path']}")
    if hasattr(torch.ops._C_ascend, "kda_gate_cumsum_vllm"):
        raise RuntimeError("An unverified registration already exists in this process")
    vendors = list((install / "vendors").glob("*/op_api/lib/libcust_opapi.so"))
    if len(vendors) != 1:
        raise RuntimeError(
            f"Expected one installed reference API library, found {vendors}"
        )
    vendor = vendors[0].parents[2]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    npu = Path(torch_npu.__file__).parent
    include = npu / "include"
    extra = {str(include), str(npu.parent), str(cann / "include"), str(source / "csrc")}
    for name in ("NPUBridge.h", "NPUStorageImpl.h"):
        extra.update(str(p.parent) for p in include.rglob(name))
    build.mkdir(parents=True, exist_ok=True)
    library = load(
        name="kda_chunk_ascendc_reference",
        sources=[
            str(Path(__file__).with_name("ascend_reference_binding.cpp")),
            str(source / "csrc/aclnn_torch_adapter/NPUBridge.cpp"),
            str(source / "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp"),
        ],
        extra_include_paths=sorted(extra),
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[
            f"-L{npu / 'lib'}",
            "-ltorch_npu",
            f"-L{cann / 'lib64'}",
            "-lascendcl",
        ],
        build_directory=str(build),
        with_cuda=False,
        is_python_module=False,
        verbose=True,
    )
    library = Path(library)
    return torch.ops._C_ascend.kda_gate_cumsum_vllm, {
        "commit": revision,
        "binding": str(library),
        "binding_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "native_api": str(vendors[0]),
        "native_api_sha256": hashlib.sha256(vendors[0].read_bytes()).hexdigest(),
        "resolved_entrypoints": _verify_api_resolution(library, vendors[0]),
        "kernel_artifacts": [
            {
                "path": str(p.relative_to(vendor)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted((vendor / "op_impl/ai_core/tbe/kernel").rglob("*"))
            if p.is_file()
        ],
        "source_tree_unmodified": True,
        "build_scope": "Single native operator; unchanged upstream host adapter and ACLNN/kernel",
        "schema": str(torch.ops._C_ascend.kda_gate_cumsum_vllm.default._schema),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--install", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, report = load_reference(args.source, args.install, args.build)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
