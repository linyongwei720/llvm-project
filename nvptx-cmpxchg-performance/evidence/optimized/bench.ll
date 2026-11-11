target triple = "nvptx64-nvidia-cuda"
@shared_slots = internal addrspace(3) global [128 x i32] undef, align 16
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define i32 @cas_guard(ptr %p, i32 %expected) noinline {
  %desired = add i32 %expected, 1
  %pair = cmpxchg ptr %p, i32 %expected, i32 %desired monotonic monotonic, align 4
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

define ptx_kernel void @bench_guard(ptr addrspace(1) %scratch,
                                        ptr addrspace(1) %out,
                                        i32 %iterations, i32 %space) {
entry:
  %slot = alloca i32, align 4
  store i32 0, ptr %slot, align 4
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
  %odd_bits = and i32 %tid, 1
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
  %p = select i1 %want_local, ptr %slot, ptr %nonlocal
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %expected = phi i32 [ 0, %entry ], [ %next_expected, %loop ]
  %old = call i32 @cas_guard(ptr %p, i32 %expected)
  %next_expected = add i32 %old, 1
  %next_i = add i32 %i, 1
  %again = icmp ult i32 %next_i, %iterations
  br i1 %again, label %loop, label %exit
exit:
  %o = getelementptr i32, ptr addrspace(1) %out, i64 %id64
  store i32 %next_expected, ptr addrspace(1) %o, align 4
  ret void
}


define i32 @cas_excluded(ptr %p, i32 %expected) noinline {
  %desired = add i32 %expected, 1
  %pair = cmpxchg ptr %p, i32 %expected, i32 %desired monotonic monotonic, align 4, !noalias.addrspace !0
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

define ptx_kernel void @bench_excluded(ptr addrspace(1) %scratch,
                                        ptr addrspace(1) %out,
                                        i32 %iterations, i32 %space) {
entry:
  %slot = alloca i32, align 4
  store i32 0, ptr %slot, align 4
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
  %odd_bits = and i32 %tid, 1
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
  %p = select i1 %want_local, ptr %slot, ptr %nonlocal
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %expected = phi i32 [ 0, %entry ], [ %next_expected, %loop ]
  %old = call i32 @cas_excluded(ptr %p, i32 %expected)
  %next_expected = add i32 %old, 1
  %next_i = add i32 %i, 1
  %again = icmp ult i32 %next_i, %iterations
  br i1 %again, label %loop, label %exit
exit:
  %o = getelementptr i32, ptr addrspace(1) %out, i64 %id64
  store i32 %next_expected, ptr addrspace(1) %o, align 4
  ret void
}

!0 = !{i32 5, i32 6}
