#!/usr/bin/env python3
"""Measure the dynamic local check against a nonlocal-metadata control."""

import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'out' / 'perf'
LLC = Path('/home/linyongwei/.cache/llvm-nvptx-pr/build/bin/llc')
CUDA = Path('/usr/local/cuda')


def make_ir():
    parts = ['target triple = "nvptx64-nvidia-cuda"',
             '@shared_slots = internal addrspace(3) global [128 x i32] undef, align 16',
             'declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()',
             'declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()',
             'declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()']
    for variant in ['guard', 'excluded']:
        metadata = '' if variant == 'guard' else ', !noalias.addrspace !0'
        parts.append(f'''
define i32 @cas_{variant}(ptr %p, i32 %expected) noinline {{
  %desired = add i32 %expected, 1
  %pair = cmpxchg ptr %p, i32 %expected, i32 %desired monotonic monotonic, align 4{metadata}
  %old = extractvalue {{ i32, i1 }} %pair, 0
  ret i32 %old
}}

define ptx_kernel void @bench_{variant}(ptr addrspace(1) %scratch,
                                        ptr addrspace(1) %out,
                                        i32 %iterations, i32 %space) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %bid, %ntid
  %id = add i32 %base, %tid
  %id64 = zext i32 %id to i64
  %tid64 = zext i32 %tid to i64
  %g = getelementptr i32, ptr addrspace(1) %scratch, i64 %id64
  %s = getelementptr [128 x i32], ptr addrspace(3) @shared_slots, i64 0, i64 %tid64
  store i32 0, ptr addrspace(1) %g, align 4
  store i32 0, ptr addrspace(3) %s, align 4
  %gp = addrspacecast ptr addrspace(1) %g to ptr
  %sp = addrspacecast ptr addrspace(3) %s to ptr
  %use_shared = icmp eq i32 %space, 1
  %p = select i1 %use_shared, ptr %sp, ptr %gp
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %expected = phi i32 [ 0, %entry ], [ %next_expected, %loop ]
  %old = call i32 @cas_{variant}(ptr %p, i32 %expected)
  %next_expected = add i32 %old, 1
  %next_i = add i32 %i, 1
  %again = icmp ult i32 %next_i, %iterations
  br i1 %again, label %loop, label %exit
exit:
  %o = getelementptr i32, ptr addrspace(1) %out, i64 %id64
  store i32 %next_expected, ptr addrspace(1) %o, align 4
  ret void
}}
''')
    parts.append('!0 = !{i32 5, i32 6}')
    return '\n'.join(parts) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(argv, name):
        argv = [str(x) for x in argv]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        (OUT / (name + '.stdout.txt')).write_text(result.stdout)
        (OUT / (name + '.stderr.txt')).write_text(result.stderr)
        commands.append({'argv': argv, 'exit': result.returncode})
        (OUT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{name}: {result.stderr}')
        return result.stdout

    (OUT / 'bench.ll').write_text(make_ir())
    run([LLC, OUT / 'bench.ll', '-O2', '-mcpu=sm_110', '-mattr=+ptx91',
         '-verify-machineinstrs', '-o', OUT / 'bench.ptx'], 'llc')
    run([CUDA / 'bin/ptxas', OUT / 'bench.ptx', '-arch=sm_110', '-O3', '-v',
         '-o', OUT / 'bench.cubin'], 'ptxas')
    run([CUDA / 'bin/nvdisasm', OUT / 'bench.cubin'], 'sass')
    run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
         '-I' + str(CUDA / 'include'), HERE / 'bench.cpp', '-lcuda',
         '-o', OUT / 'bench'], 'build-host')
    result = json.loads(run([OUT / 'bench', OUT / 'bench.cubin'], 'bench'))
    result['method'] = ('Same patched compiler and cubin; generic cmpxchg with '
                        'a runtime local check versus the identical source '
                        'with !noalias.addrspace excluding local. All measured '
                        'addresses are global/shared. No local-memory baseline '
                        'or real-application performance is measured.')
    result['llc_sha256'] = hashlib.sha256(LLC.read_bytes()).hexdigest()
    result['inputs_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [HERE / 'run.py', HERE / 'bench.cpp',
                                         OUT / 'bench.ll']}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result['cases']:
        print(json.dumps({k: v for k, v in row.items() if k != 'samples'},
                         ensure_ascii=False))
    print('Evidence:', OUT / 'results.json')


if __name__ == '__main__':
    main()
