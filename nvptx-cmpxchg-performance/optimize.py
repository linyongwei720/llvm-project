#!/usr/bin/env python3
"""Compare actual compiler candidates, including mixed local/nonlocal warps."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import run as baseline


HERE = Path(__file__).resolve().parent
CUDA = Path('/usr/local/cuda')


def replace_once_or_all(text, old, new, count=None):
    if old not in text:
        raise ValueError('Missing source template: ' + old[:70])
    return text.replace(old, new) if count is None else text.replace(old, new, count)


def make_ir():
    text = baseline.make_ir()
    text = replace_once_or_all(text, 'entry:\n  %tid =',
                              'entry:\n  %slot = alloca i32, align 4\n'
                              '  store i32 0, ptr %slot, align 4\n  %tid =')
    text = replace_once_or_all(text,
        '  %p = select i1 %use_shared, ptr %sp, ptr %gp', '''  %odd_bits = and i32 %tid, 1
  %odd = icmp ne i32 %odd_bits, 0
  %local_only = icmp eq i32 %space, 2
  %mix_global_local = icmp eq i32 %space, 3
  %mix_shared_local = icmp eq i32 %space, 4
  %mix_global_shared = icmp eq i32 %space, 5
  %mixed_local = or i1 %mix_global_local, %mix_shared_local
  %odd_local = and i1 %mixed_local, %odd
  %want_local = or i1 %local_only, %odd_local
  %odd_shared = and i1 %mix_global_shared, %odd
  %any_shared = or i1 %use_shared, %mix_shared_local
  %want_shared = or i1 %any_shared, %odd_shared
  %nonlocal = select i1 %want_shared, ptr %sp, ptr %gp
  %p = select i1 %want_local, ptr %slot, ptr %nonlocal''')
    return text


def make_driver():
    text = (HERE / 'bench.cpp').read_text()
    text = replace_once_or_all(text, 'if (argc != 2)',
                              'if (argc != 3 && argc != 4)')
    text = replace_once_or_all(text,
        '  CUfunction functions[2];\n'
        '  check(cuModuleGetFunction(&functions[0], module, "bench_excluded"), "control");\n'
        '  check(cuModuleGetFunction(&functions[1], module, "bench_guard"), "guard");',
        '''  CUmodule candidate;
  check(cuModuleLoad(&candidate, argv[2]), "load candidate");
  const char *control_name = argc == 4 ? argv[3] : "bench_guard";
  const bool native_control = argc == 4;
  CUfunction functions[2];
  check(cuModuleGetFunction(&functions[0], module, control_name), "control");
  check(cuModuleGetFunction(&functions[1], candidate, "bench_guard"), "guard");
  const char *space_names[] = {"global", "shared", "local", "mixed-global-local",
                               "mixed-shared-local", "mixed-global-shared"};''')
    text = replace_once_or_all(text,
        'for (unsigned space = 0; space != 2; ++space)',
        'for (unsigned space = 0; space != (native_control ? 2u : 6u); ++space)')
    text = replace_once_or_all(text,
        '    for (unsigned mode = 0; mode != 2; ++mode) {',
        '    for (unsigned mode = 0; mode != 2; ++mode) {\n'
        '      if (space >= 3 && mode == 0)\n        continue;')
    text = replace_once_or_all(text, 'space ? "shared" : "global",',
                              'space_names[space],')
    text = replace_once_or_all(text,
        '  check(cuModuleUnload(module), "unload");',
        '  check(cuModuleUnload(candidate), "unload candidate");\n'
        '  check(cuModuleUnload(module), "unload");')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(argv, name):
        argv = [str(x) for x in argv]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=90)
        (out / (name + '.stdout.txt')).write_text(result.stdout)
        (out / (name + '.stderr.txt')).write_text(result.stderr)
        commands.append({'argv': argv, 'exit': result.returncode})
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{name}: {result.stderr}')
        return result.stdout

    (out / 'bench.ll').write_text(make_ir())
    (out / 'compare.cpp').write_text(make_driver())
    for variant, llc in [('original', args.original), ('candidate', args.candidate)]:
        run([llc, out / 'bench.ll', '-O2', '-mcpu=sm_110', '-mattr=+ptx91',
             '-verify-machineinstrs', '-o', out / (variant + '.ptx')], variant + '-llc')
        run([CUDA / 'bin/ptxas', out / (variant + '.ptx'), '-arch=sm_110',
             '-O3', '-v', '-o', out / (variant + '.cubin')], variant + '-ptxas')
        run([CUDA / 'bin/nvdisasm', out / (variant + '.cubin')], variant + '-sass')
    run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
         '-I' + str(CUDA / 'include'), out / 'compare.cpp', '-lcuda',
         '-o', out / 'compare'], 'host-build')
    comparisons = [
        ('native-vs-original', 'original', 'original', True),
        ('original-vs-candidate', 'original', 'candidate', False),
        ('native-vs-candidate', 'candidate', 'candidate', True),
    ]
    summary = {'method': 'Real llc output; identical IR and launch configurations; '
                        '21 randomized paired samples after six warm-up pairs. '
                        'Native-metadata controls only run on global/shared addresses.',
               'llc_sha256': {v: hashlib.sha256(p.read_bytes()).hexdigest()
                              for v, p in [('original', args.original),
                                           ('candidate', args.candidate)]},
               'comparisons': {}}
    for name, left, right, native in comparisons:
        command = [out / 'compare', out / (left + '.cubin'), out / (right + '.cubin')]
        if native:
            command.append('bench_excluded')
        result = json.loads(run(command, name))
        (out / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        rows = [{k: v for k, v in case.items() if k != 'samples'}
                for case in result['cases']]
        summary['comparisons'][name] = rows
        print(name, json.dumps(rows), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
