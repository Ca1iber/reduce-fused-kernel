; ModuleID = '/data/TileOPs-Metax/profiles/v017/codegen/k2_sf0_s2_mode2.cu'
source_filename = "/data/TileOPs-Metax/profiles/v017/codegen/k2_sf0_s2_mode2.cu"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "mxc-metax-macahca"

%struct.__maca_bfloat16.1 = type { i16 }
%struct.mcDevMallocInfo.0 = type { i32, i32, ptr }

@stage_x = external protected addrspace(3) global [0 x %struct.__maca_bfloat16.1], align 1024
@mcDeviceMemoryInfo = weak protected addrspace(1) externally_initialized global [1 x %struct.mcDevMallocInfo.0] zeroinitializer, align 8
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @mcDeviceMemoryInfo to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress noreturn nounwind
define weak void @__cxa_pure_virtual() local_unnamed_addr #0 {
entry:
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: mustprogress noreturn nounwind
define weak void @__cxa_deleted_virtual() local_unnamed_addr #0 {
entry:
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind
define weak protected void @__mcImplicitDeviceSynchronize() local_unnamed_addr #2 {
entry:
  %0 = tail call ptr addrspace(4) @llvm.mxc.implicitarg.ptr()
  %arrayidx.i.i = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %1 = load i64, ptr addrspace(4) %arrayidx.i.i, align 8, !tbaa !2
  %2 = inttoptr i64 %1 to ptr
  %arrayidx.i32.i = getelementptr inbounds i8, ptr addrspace(4) %0, i64 40
  %3 = load i64, ptr addrspace(4) %arrayidx.i32.i, align 8, !tbaa !2
  %4 = inttoptr i64 %3 to ptr
  %current_counter.i = getelementptr inbounds i8, ptr %4, i64 12
  %thread_counter.i = getelementptr inbounds i8, ptr %4, i64 16
  %5 = load ptr, ptr %thread_counter.i, align 8, !tbaa !6
  %cmp.not.i = icmp eq ptr %5, null
  br i1 %cmp.not.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %entry
  %6 = tail call align 4 dereferenceable(64) ptr addrspace(4) @llvm.mxc.dispatch.ptr()
  %7 = getelementptr inbounds i8, ptr addrspace(4) %6, i64 12
  %8 = load i32, ptr addrspace(4) %7, align 4, !range !13, !invariant.load !14
  %9 = getelementptr inbounds i8, ptr addrspace(4) %6, i64 4
  %10 = load i32, ptr addrspace(4) %9, align 4, !invariant.load !14
  %conv.i14.i.i = and i32 %10, 65535
  %div.i15.i.i = udiv i32 %8, %conv.i14.i.i
  %mul.i16.i.i = mul i32 %div.i15.i.i, %conv.i14.i.i
  %cmp.i17.i.i = icmp ugt i32 %8, %mul.i16.i.i
  %conv2.i18.i.i = zext i1 %cmp.i17.i.i to i32
  %add.i19.i.i = add nuw i32 %div.i15.i.i, %conv2.i18.i.i
  %11 = getelementptr inbounds i8, ptr addrspace(4) %6, i64 16
  %12 = load i32, ptr addrspace(4) %11, align 4, !range !13, !invariant.load !14
  %13 = lshr i32 %10, 16
  %div.i.i.i = udiv i32 %12, %13
  %mul.i.i.i = mul i32 %div.i.i.i, %13
  %cmp.i.i.i = icmp ugt i32 %12, %mul.i.i.i
  %conv2.i.i.i = zext i1 %cmp.i.i.i to i32
  %add.i.i.i = add nuw i32 %div.i.i.i, %conv2.i.i.i
  %14 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.z(), !range !15
  %mul.i.i = mul i32 %add.i.i.i, %14
  %15 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.y(), !range !15
  %mul320.i.i = add i32 %mul.i.i, %15
  %add.i.i = mul i32 %mul320.i.i, %add.i19.i.i
  %16 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.x(), !range !15
  %add8.i.i = add i32 %add.i.i, %16
  %conv.i.i = zext i32 %add8.i.i to i64
  %arrayidx.i = getelementptr inbounds i32, ptr %5, i64 %conv.i.i
  %17 = tail call i1 @llvm.mxc.is.private(ptr nonnull %arrayidx.i)
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.body.i, %if.then.i
  br i1 %17, label %if.then.i.i, label %if.end.i.i

if.then.i.i:                                      ; preds = %while.cond.i
  %18 = load i32, ptr %arrayidx.i, align 4, !tbaa !16
  br label %_Z9atomicAddPii.exit.i

if.end.i.i:                                       ; preds = %while.cond.i
  %19 = atomicrmw or ptr %arrayidx.i, i32 0 syncscope("device-one-as") monotonic, align 4
  br label %_Z9atomicAddPii.exit.i

_Z9atomicAddPii.exit.i:                           ; preds = %if.end.i.i, %if.then.i.i
  %retval.0.i.i = phi i32 [ %18, %if.then.i.i ], [ %19, %if.end.i.i ]
  %cmp6.i = icmp sgt i32 %retval.0.i.i, 0
  br i1 %cmp6.i, label %while.body.i, label %if.end.i

while.body.i:                                     ; preds = %_Z9atomicAddPii.exit.i
  tail call void @llvm.mxc.sleep(i32 1)
  br label %while.cond.i, !llvm.loop !17

if.end.i:                                         ; preds = %_Z9atomicAddPii.exit.i, %entry
  %20 = tail call i1 @llvm.mxc.is.private(ptr nonnull %current_counter.i)
  br i1 %20, label %if.then.i36.i, label %if.end.i34.i

if.then.i36.i:                                    ; preds = %if.end.i
  %21 = load i32, ptr %current_counter.i, align 4, !tbaa !16
  %add.i37.i = add nsw i32 %21, -1
  store i32 %add.i37.i, ptr %current_counter.i, align 4, !tbaa !16
  br label %_Z9atomicAddPii.exit38.i

if.end.i34.i:                                     ; preds = %if.end.i
  %22 = atomicrmw add ptr %current_counter.i, i32 -1 syncscope("device-one-as") monotonic, align 4
  br label %_Z9atomicAddPii.exit38.i

_Z9atomicAddPii.exit38.i:                         ; preds = %if.end.i34.i, %if.then.i36.i
  %retval.0.i35.i = phi i32 [ %21, %if.then.i36.i ], [ %22, %if.end.i34.i ]
  fence syncscope("device") seq_cst
  %call_deep.i = getelementptr inbounds i8, ptr %4, i64 4
  %23 = load i32, ptr %call_deep.i, align 4, !tbaa !19
  %cmp8.i = icmp eq i32 %23, 0
  br i1 %cmp8.i, label %if.then9.i, label %if.else.i

if.then9.i:                                       ; preds = %_Z9atomicAddPii.exit38.i
  %cmp10.i = icmp slt i32 %retval.0.i35.i, 2
  br i1 %cmp10.i, label %if.then11.i, label %__mcImplicitDeviceSynchronizeImpl.exit

if.then11.i:                                      ; preds = %if.then9.i
  tail call void @llvm.mxc.sleep(i32 30)
  br label %__mcImplicitDeviceSynchronizeImpl.exit

if.else.i:                                        ; preds = %_Z9atomicAddPii.exit38.i
  %parent_wrap13.i = getelementptr inbounds i8, ptr %4, i64 24
  %24 = load ptr, ptr %parent_wrap13.i, align 8, !tbaa !20
  %25 = load i32, ptr %4, align 8, !tbaa !21
  %thread_counter15.i = getelementptr inbounds i8, ptr %24, i64 16
  %26 = load ptr, ptr %thread_counter15.i, align 8, !tbaa !6
  %idxprom.i = zext i32 %25 to i64
  %arrayidx16.i = getelementptr inbounds i32, ptr %26, i64 %idxprom.i
  %cmp17.i = icmp slt i32 %retval.0.i35.i, 2
  br i1 %cmp17.i, label %if.then18.i, label %if.end20.i

if.then18.i:                                      ; preds = %if.else.i
  %mqlwrap_slots.i = getelementptr inbounds i8, ptr %2, i64 24
  %27 = load ptr, ptr %mqlwrap_slots.i, align 8, !tbaa !22
  %sub.ptr.rhs.cast.i = ptrtoint ptr %27 to i64
  %sub.ptr.sub.i = sub i64 %3, %sub.ptr.rhs.cast.i
  %sub.ptr.div.i = sdiv exact i64 %sub.ptr.sub.i, 96
  %mqlwrap_slot_mask.i = getelementptr inbounds i8, ptr %2, i64 80
  %28 = load i64, ptr %mqlwrap_slot_mask.i, align 8, !tbaa !24
  %29 = inttoptr i64 %28 to ptr
  %idx.ext.i.i = and i64 %sub.ptr.div.i, 4294967295
  %add.ptr.i.i = getelementptr inbounds i32, ptr %29, i64 %idx.ext.i.i
  %30 = tail call i1 @llvm.mxc.is.private(ptr %add.ptr.i.i)
  br i1 %30, label %if.then.i.i.i, label %if.end.i.i.i

if.then.i.i.i:                                    ; preds = %if.then18.i
  %31 = load i32, ptr %add.ptr.i.i, align 4, !tbaa !16
  %and.i.i.i = and i32 %31, 2147483647
  store i32 %and.i.i.i, ptr %add.ptr.i.i, align 4, !tbaa !16
  br label %if.end20.i

if.end.i.i.i:                                     ; preds = %if.then18.i
  %32 = atomicrmw and ptr %add.ptr.i.i, i32 2147483647 syncscope("device-one-as") monotonic, align 4
  br label %if.end20.i

if.end20.i:                                       ; preds = %if.end.i.i.i, %if.then.i.i.i, %if.else.i
  %33 = tail call i1 @llvm.mxc.is.private(ptr %arrayidx16.i)
  br i1 %33, label %if.then.i41.i, label %if.end.i39.i

if.then.i41.i:                                    ; preds = %if.end20.i
  %34 = load i32, ptr %arrayidx16.i, align 4, !tbaa !16
  %add.i42.i = add nsw i32 %34, -1
  store i32 %add.i42.i, ptr %arrayidx16.i, align 4, !tbaa !16
  br label %__mcImplicitDeviceSynchronizeImpl.exit

if.end.i39.i:                                     ; preds = %if.end20.i
  %35 = atomicrmw add ptr %arrayidx16.i, i32 -1 syncscope("device-one-as") monotonic, align 4
  br label %__mcImplicitDeviceSynchronizeImpl.exit

__mcImplicitDeviceSynchronizeImpl.exit:           ; preds = %if.then9.i, %if.then11.i, %if.then.i41.i, %if.end.i39.i
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind willreturn
define protected metaxgpu_kernel void @reduce_fused_kernel_kernel(ptr addrspace(1) noalias nocapture noundef writeonly %out.coerce, ptr addrspace(1) noalias nocapture noundef readonly %token_topk_to_pos.coerce, ptr addrspace(1) noalias nocapture noundef readonly %topk_weights.coerce, ptr addrspace(4) noalias nocapture noundef %x.coerce, i32 noundef %num_tokens) local_unnamed_addr #3 {
entry:
  %0 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.x(), !range !15
  %1 = shl nuw i32 %0, 1
  %mul = zext i32 %1 to i64
  %add.ptr6 = getelementptr inbounds float, ptr addrspace(1) %topk_weights.coerce, i64 %mul
  %2 = load float, ptr addrspace(1) %add.ptr6, align 8
  %add.ptr6.sroa_idx = getelementptr inbounds i8, ptr addrspace(1) %add.ptr6, i64 4
  %3 = load float, ptr addrspace(1) %add.ptr6.sroa_idx, align 4
  %add.ptr12 = getelementptr inbounds i32, ptr addrspace(1) %token_topk_to_pos.coerce, i64 %mul
  %4 = load i64, ptr addrspace(1) %add.ptr12, align 8
  %topk_to_pos_local.sroa.5.0.extract.shift = lshr i64 %4, 32
  %5 = and i64 %4, 2147483648
  %cmp = icmp eq i64 %5, 0
  br i1 %cmp, label %if.then, label %entry.if.end_crit_edge

entry.if.end_crit_edge:                           ; preds = %entry
  %.pre = tail call range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !25
  %.pre176 = shl nuw nsw i32 %.pre, 2
  %.pre177 = tail call range(i32 0, 2147483647) i32 @llvm.mxc.block.id.y(), !range !15
  %.pre179 = zext nneg i32 %.pre177 to i64
  %.pre180 = zext nneg i32 %.pre176 to i64
  %.pre181 = shl nuw nsw i64 %.pre179, 10
  br label %if.end

if.then:                                          ; preds = %entry
  %conv18 = and i64 %4, 2147483647
  %.idx = mul nuw nsw i64 %conv18, 14336
  %6 = getelementptr inbounds i8, ptr addrspace(4) %x.coerce, i64 %.idx
  %7 = tail call noundef range(i32 0, 2147483647) i32 @llvm.mxc.block.id.y(), !range !15
  %conv21 = zext nneg i32 %7 to i64
  %.idx154 = shl nuw nsw i64 %conv21, 10
  %8 = getelementptr inbounds i8, ptr addrspace(4) %6, i64 %.idx154
  %9 = tail call noundef range(i32 0, 1024) i32 @llvm.mxc.thread.id.x(), !range !25
  %mul16 = shl nuw nsw i32 %9, 2
  %idxprom = zext nneg i32 %mul16 to i64
  %arrayidx27 = getelementptr inbounds %struct.__maca_bfloat16.1, ptr addrspace(4) %8, i64 %idxprom
  %10 = addrspacecast ptr addrspace(4) %arrayidx27 to ptr addrspace(1)
  %arrayidx17 = getelementptr inbounds [0 x %struct.__maca_bfloat16.1], ptr addrspace(3) @stage_x, i32 0, i32 %mul16
  %11 = tail call noundef <2 x i32> @llvm.mxc.ldg.predicator.bsm.v2i32(ptr addrspace(3) %arrayidx17, ptr addrspace(1) %10, i32 0, i64 -1, i1 false, i1 true, i1 false, i1 false), !alias.scope !26, !call_argsrelate !30
  br label %if.end

if.end:                                           ; preds = %entry.if.end_crit_edge, %if.then
  %.idx158.pre-phi = phi i64 [ %.pre181, %entry.if.end_crit_edge ], [ %.idx154, %if.then ]
  %mul58.pre-phi = phi i64 [ %.pre180, %entry.if.end_crit_edge ], [ %idxprom, %if.then ]
  %conv53.pre-phi = phi i64 [ %.pre179, %entry.if.end_crit_edge ], [ %conv21, %if.then ]
  %mul45.pre-phi = phi i32 [ %.pre176, %entry.if.end_crit_edge ], [ %mul16, %if.then ]
  %.pre-phi = phi i32 [ %.pre, %entry.if.end_crit_edge ], [ %9, %if.then ]
  %tickets.sroa.0.0 = phi <2 x i32> [ undef, %entry.if.end_crit_edge ], [ %11, %if.then ]
  %add47 = add nuw nsw i32 %mul45.pre-phi, 512
  %arrayidx49 = getelementptr inbounds [0 x %struct.__maca_bfloat16.1], ptr addrspace(3) @stage_x, i32 0, i32 %add47
  %invariant.gep = getelementptr inbounds i8, ptr addrspace(4) %x.coerce, i64 %.idx158.pre-phi
  %invariant.gep170 = getelementptr inbounds %struct.__maca_bfloat16.1, ptr addrspace(4) %invariant.gep, i64 %mul58.pre-phi
  %add.ptr73.idx = shl nuw nsw i32 %.pre-phi, 3
  %invariant.gep172 = getelementptr inbounds i8, ptr addrspace(3) @stage_x, i32 %add.ptr73.idx
  br i1 %cmp, label %if.then34, label %if.end37

if.then34:                                        ; preds = %if.end
  tail call void @llvm.mxc.barrier.and.wait2(i32 1, <2 x i32> %tickets.sroa.0.0)
  br label %if.end37

if.end37:                                         ; preds = %if.then34, %if.end
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  %cmp41 = icmp sgt i64 %4, -1
  br i1 %cmp41, label %if.then42, label %if.end64

if.then42:                                        ; preds = %if.end37
  %.idx157 = mul nuw nsw i64 %topk_to_pos_local.sroa.5.0.extract.shift, 14336
  %gep171 = getelementptr inbounds i8, ptr addrspace(4) %invariant.gep170, i64 %.idx157
  %12 = addrspacecast ptr addrspace(4) %gep171 to ptr addrspace(1)
  %13 = tail call noundef <2 x i32> @llvm.mxc.ldg.predicator.bsm.v2i32(ptr addrspace(3) %arrayidx49, ptr addrspace(1) %12, i32 0, i64 -1, i1 false, i1 true, i1 false, i1 false), !alias.scope !31, !call_argsrelate !30
  br label %if.end64

if.end64:                                         ; preds = %if.end37, %if.then42
  %tickets.sroa.4.0 = phi <2 x i32> [ %13, %if.then42 ], [ undef, %if.end37 ]
  br i1 %cmp, label %if.then66, label %if.end121

if.then66:                                        ; preds = %if.end64
  %14 = load i64, ptr addrspace(3) %invariant.gep172, align 8
  %v__1.sroa.0.0.extract.trunc = trunc i64 %14 to i32
  %v__1.sroa.4.0.extract.shift = lshr i64 %14, 32
  %v__1.sroa.4.0.extract.trunc = trunc nuw i64 %v__1.sroa.4.0.extract.shift to i32
  %agg.tmp.sroa.2.0.extract.shift.i = and i32 %v__1.sroa.0.0.extract.trunc, -65536
  %conv.i.i.i = shl i32 %v__1.sroa.0.0.extract.trunc, 16
  %15 = bitcast i32 %conv.i.i.i to float
  %16 = bitcast i32 %agg.tmp.sroa.2.0.extract.shift.i to float
  %agg.tmp.sroa.2.0.extract.shift.i160 = and i32 %v__1.sroa.4.0.extract.trunc, -65536
  %conv.i.i.i161 = shl i32 %v__1.sroa.4.0.extract.trunc, 16
  %17 = bitcast i32 %conv.i.i.i161 to float
  %18 = bitcast i32 %agg.tmp.sroa.2.0.extract.shift.i160 to float
  %mul92 = fmul contract float %2, %15
  %mul95 = fmul contract float %2, %16
  %mul98 = fmul contract float %2, %17
  %mul101 = fmul contract float %2, %18
  %add105 = fadd contract float %mul92, 0.000000e+00
  %add109 = fadd contract float %mul95, 0.000000e+00
  %add113 = fadd contract float %mul98, 0.000000e+00
  %add117 = fadd contract float %mul101, 0.000000e+00
  br label %if.end121

if.end121:                                        ; preds = %if.then66, %if.end64
  %reduced_fragment.sroa.0.1 = phi float [ %add105, %if.then66 ], [ 0.000000e+00, %if.end64 ]
  %reduced_fragment.sroa.6.1 = phi float [ %add109, %if.then66 ], [ 0.000000e+00, %if.end64 ]
  %reduced_fragment.sroa.9.1 = phi float [ %add113, %if.then66 ], [ 0.000000e+00, %if.end64 ]
  %reduced_fragment.sroa.12.1 = phi float [ %add117, %if.then66 ], [ 0.000000e+00, %if.end64 ]
  %cmp33.1 = icmp sgt i64 %4, -1
  br i1 %cmp33.1, label %if.then34.1, label %if.end37.1

if.then34.1:                                      ; preds = %if.end121
  tail call void @llvm.mxc.barrier.and.wait2(i32 1, <2 x i32> %tickets.sroa.4.0)
  br label %if.end37.1

if.end37.1:                                       ; preds = %if.then34.1, %if.end121
  fence syncscope("block") release
  tail call void @llvm.mxc.barrier()
  fence syncscope("block") acquire
  br i1 %cmp33.1, label %if.then66.1, label %if.end121.1

if.then66.1:                                      ; preds = %if.end37.1
  %gep.1 = getelementptr inbounds i8, ptr addrspace(3) %invariant.gep172, i32 1024
  %19 = load i64, ptr addrspace(3) %gep.1, align 8
  %v__1.sroa.0.0.extract.trunc.1 = trunc i64 %19 to i32
  %v__1.sroa.4.0.extract.shift.1 = lshr i64 %19, 32
  %v__1.sroa.4.0.extract.trunc.1 = trunc nuw i64 %v__1.sroa.4.0.extract.shift.1 to i32
  %agg.tmp.sroa.2.0.extract.shift.i.1 = and i32 %v__1.sroa.0.0.extract.trunc.1, -65536
  %conv.i.i.i.1 = shl i32 %v__1.sroa.0.0.extract.trunc.1, 16
  %20 = bitcast i32 %conv.i.i.i.1 to float
  %21 = bitcast i32 %agg.tmp.sroa.2.0.extract.shift.i.1 to float
  %agg.tmp.sroa.2.0.extract.shift.i160.1 = and i32 %v__1.sroa.4.0.extract.trunc.1, -65536
  %conv.i.i.i161.1 = shl i32 %v__1.sroa.4.0.extract.trunc.1, 16
  %22 = bitcast i32 %conv.i.i.i161.1 to float
  %23 = bitcast i32 %agg.tmp.sroa.2.0.extract.shift.i160.1 to float
  %mul92.1 = fmul contract float %3, %20
  %mul95.1 = fmul contract float %3, %21
  %mul98.1 = fmul contract float %3, %22
  %mul101.1 = fmul contract float %3, %23
  %add105.1 = fadd contract float %reduced_fragment.sroa.0.1, %mul92.1
  %add109.1 = fadd contract float %reduced_fragment.sroa.6.1, %mul95.1
  %add113.1 = fadd contract float %reduced_fragment.sroa.9.1, %mul98.1
  %add117.1 = fadd contract float %reduced_fragment.sroa.12.1, %mul101.1
  br label %if.end121.1

if.end121.1:                                      ; preds = %if.then66.1, %if.end37.1
  %reduced_fragment.sroa.0.1.1 = phi float [ %add105.1, %if.then66.1 ], [ %reduced_fragment.sroa.0.1, %if.end37.1 ]
  %reduced_fragment.sroa.6.1.1 = phi float [ %add109.1, %if.then66.1 ], [ %reduced_fragment.sroa.6.1, %if.end37.1 ]
  %reduced_fragment.sroa.9.1.1 = phi float [ %add113.1, %if.then66.1 ], [ %reduced_fragment.sroa.9.1, %if.end37.1 ]
  %reduced_fragment.sroa.12.1.1 = phi float [ %add117.1, %if.then66.1 ], [ %reduced_fragment.sroa.12.1, %if.end37.1 ]
  %conv125 = zext nneg i32 %0 to i64
  %.idx155 = mul nuw nsw i64 %conv125, 28672
  %24 = getelementptr inbounds i8, ptr addrspace(1) %out.coerce, i64 %.idx155
  %.idx156 = shl nuw nsw i64 %conv53.pre-phi, 11
  %25 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 %.idx156
  %add.ptr135 = getelementptr inbounds float, ptr addrspace(1) %25, i64 %mul58.pre-phi
  store float %reduced_fragment.sroa.0.1.1, ptr addrspace(1) %add.ptr135, align 16, !tbaa !35
  %reduced_fragment.sroa.6.0.add.ptr135.sroa_idx = getelementptr inbounds i8, ptr addrspace(1) %add.ptr135, i64 4
  store float %reduced_fragment.sroa.6.1.1, ptr addrspace(1) %reduced_fragment.sroa.6.0.add.ptr135.sroa_idx, align 4, !tbaa !35
  %reduced_fragment.sroa.9.0.add.ptr135.sroa_idx = getelementptr inbounds i8, ptr addrspace(1) %add.ptr135, i64 8
  store float %reduced_fragment.sroa.9.1.1, ptr addrspace(1) %reduced_fragment.sroa.9.0.add.ptr135.sroa_idx, align 8, !tbaa !35
  %reduced_fragment.sroa.12.0.add.ptr135.sroa_idx = getelementptr inbounds i8, ptr addrspace(1) %add.ptr135, i64 12
  store float %reduced_fragment.sroa.12.1.1, ptr addrspace(1) %reduced_fragment.sroa.12.0.add.ptr135.sroa_idx, align 4, !tbaa !35
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.block.id.x() #4

; Function Attrs: mustprogress nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.block.id.y() #4

; Function Attrs: mustprogress nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.block.id.z() #4

; Function Attrs: mustprogress nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.mxc.thread.id.x() #4

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @llvm.mxc.barrier() #5

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare <2 x i32> @llvm.mxc.ldg.predicator.bsm.v2i32(ptr addrspace(3) noalias nocapture, ptr addrspace(1) noalias nocapture, i32 immarg, i64, i1 immarg, i1 immarg, i1 immarg, i1 immarg) #6

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @llvm.mxc.barrier.and.wait2(i32, <2 x i32>) #5

; Function Attrs: mustprogress nofree nosync nounwind speculatable willreturn memory(none)
declare align 4 ptr addrspace(4) @llvm.mxc.implicitarg.ptr() #4

; Function Attrs: mustprogress nofree nosync nounwind speculatable willreturn memory(none)
declare align 4 ptr addrspace(4) @llvm.mxc.dispatch.ptr() #4

; Function Attrs: mustprogress nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.mxc.is.private(ptr nocapture) #4

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.mxc.sleep(i32 immarg) #7

attributes #0 = { mustprogress noreturn nounwind "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { nounwind "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" }
attributes #3 = { convergent mustprogress norecurse nounwind willreturn "disable-promote-alloca-to-bsm"="true" "disable-promote-alloca-to-vector"="false" "enable-ldg-bsm-opt"="false" "fixed-function-abi"="true" "metaxgpu-bsm-direct-address"="true" "metaxgpu-implicitarg-num-bytes"="80" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="128" "metaxgpu-min-blocks"="1" "metaxgpu-new-streg-abi"="false" "metaxgpu-pk-fma"="false" "metaxgpu-resource-usage"="false" "metaxgpu-sched-select"="default" "metaxgpu-use-dim-intrinsic"="false" "no-trapping-math"="true" "scalarize-global-loads"="true" "shfl-combine"="true" "stack-protector-buffer-size"="8" "target-cpu"="xcore1000" "target-features"="+xcore1000" "uniform-work-group-size"="true" }
attributes #4 = { mustprogress nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent mustprogress nounwind willreturn }
attributes #6 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { mustprogress nounwind willreturn }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 1}
!2 = !{!3, !3, i64 0}
!3 = !{!"long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !9, i64 16}
!7 = !{!"_ZTS21_DynamicParallMqlWrap", !8, i64 0, !8, i64 4, !8, i64 8, !8, i64 12, !9, i64 16, !9, i64 24, !10, i64 32}
!8 = !{!"int", !4, i64 0}
!9 = !{!"any pointer", !4, i64 0}
!10 = !{!"_ZTS28mxc_kernel_dispatch_packet_s", !11, i64 0, !11, i64 2, !11, i64 4, !11, i64 6, !11, i64 8, !11, i64 10, !8, i64 12, !8, i64 16, !8, i64 20, !8, i64 24, !8, i64 28, !3, i64 32, !9, i64 40, !3, i64 48, !12, i64 56}
!11 = !{!"short", !4, i64 0}
!12 = !{!"_ZTS12mxc_signal_s", !3, i64 0}
!13 = !{i32 1, i32 -2147483648}
!14 = !{}
!15 = !{i32 0, i32 2147483647}
!16 = !{!8, !8, i64 0}
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.mustprogress"}
!19 = !{!7, !8, i64 4}
!20 = !{!7, !9, i64 24}
!21 = !{!7, !8, i64 0}
!22 = !{!23, !9, i64 24}
!23 = !{!"_ZTS28_DynamicParallSchedulerParam", !3, i64 0, !9, i64 8, !9, i64 16, !9, i64 24, !3, i64 32, !3, i64 40, !3, i64 48, !3, i64 56, !3, i64 64, !3, i64 72, !3, i64 80, !3, i64 88, !3, i64 96, !8, i64 104, !8, i64 108, !8, i64 112, !8, i64 116, !8, i64 120, !8, i64 124}
!24 = !{!23, !3, i64 80}
!25 = !{i32 0, i32 1024}
!26 = !{!27, !29}
!27 = distinct !{!27, !28, !"_ZL12memcpy_asyncILi8ELb0EENSt9enable_ifIXeqT_Li8EEDv2_jE4typeEPvS4_: %dst_shared"}
!28 = distinct !{!28, !"_ZL12memcpy_asyncILi8ELb0EENSt9enable_ifIXeqT_Li8EEDv2_jE4typeEPvS4_"}
!29 = distinct !{!29, !28, !"_ZL12memcpy_asyncILi8ELb0EENSt9enable_ifIXeqT_Li8EEDv2_jE4typeEPvS4_: %src_global"}
!30 = !{i32 -1, i32 3, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!31 = !{!32, !34}
!32 = distinct !{!32, !33, !"_ZL12memcpy_asyncILi8ELb0EENSt9enable_ifIXeqT_Li8EEDv2_jE4typeEPvS4_: %dst_shared"}
!33 = distinct !{!33, !"_ZL12memcpy_asyncILi8ELb0EENSt9enable_ifIXeqT_Li8EEDv2_jE4typeEPvS4_"}
!34 = distinct !{!34, !33, !"_ZL12memcpy_asyncILi8ELb0EENSt9enable_ifIXeqT_Li8EEDv2_jE4typeEPvS4_: %src_global"}
!35 = !{!36, !36, i64 0}
!36 = !{!"float", !4, i64 0}
