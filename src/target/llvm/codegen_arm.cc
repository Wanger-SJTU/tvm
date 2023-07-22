/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_arm.cc
 * \brief ARM specific code generator
 */
#ifdef TVM_LLVM_VERSION

#include <llvm/IR/Intrinsics.h>
#include <tvm/runtime/registry.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsARM.h>
#endif
#include <llvm/Target/TargetMachine.h>

#include "codegen_cpu.h"

namespace tvm {
namespace codegen {

// ARM specific code generator, this is used as an example on
// how to override behavior llvm code generator for specific target
class CodeGenARM : public CodeGenCPU {
 public:
  CodeGenARM() = default;
  virtual ~CodeGenARM() = default;

  void InitTarget() {
    // set native vector bits.
    native_vector_bits_ = 16 * 8;
    CodeGenCPU::InitTarget();
  }
  llvm::Value* CreateIntrinsic(const CallNode* op) override;

 private:
  PrimExpr ARMPopcount(const CallNode* op);
};

llvm::Value* CodeGenARM::CreateIntrinsic(const CallNode* op) {
  if (op->op.same_as(builtin_call_llvm_intrin_) || op->op.same_as(builtin_call_llvm_pure_intrin_)) {
    llvm::Intrinsic::ID id = static_cast<llvm::Intrinsic::ID>(Downcast<IntImm>(op->args[0])->value);
    if (id == llvm::Intrinsic::ctpop) {
      PrimExpr e = ARMPopcount(op);
      return CodeGenCPU::CreateIntrinsic(e.as<CallNode>());
    }
  }
  return CodeGenCPU::CreateIntrinsic(op);
}

PrimExpr CodeGenARM::ARMPopcount(const CallNode* call) {
  using namespace tir;
  const PrimExpr& e = call->args[2];
  llvm::Intrinsic::ID ctpop_id = llvm::Intrinsic::ctpop;
  llvm::Intrinsic::ID vpaddlu_id = llvm::Intrinsic::arm_neon_vpaddlu;

  // Fallback to default llvm lowering rule if input type not a full vector or half vector length
  int total_size = call->dtype.bits() * call->dtype.lanes();
  if (!call->dtype.is_vector() || call->dtype.bits() == 8 ||
      (total_size != 128 && total_size != 64)) {
    Array<PrimExpr> vcnt_args;
    vcnt_args.push_back(IntImm(DataType::UInt(32), ctpop_id));
    vcnt_args.push_back(IntImm(DataType::UInt(32), 1));
    vcnt_args.push_back(e);
    return tir::Call(call->dtype, builtin_call_llvm_pure_intrin_, vcnt_args);
  }

  // Popcount lowering rule:
  // Reinterpret input vector as a vector of 8bit values and preform popcount
  // Pairwise add between adjacent elements and double width with vpaddlu
  // to return back to original input type

  // Dvisions are always divisible (number of bits = 64 or 128)
  DataType uint8_type = DataType(e.dtype().code(), 8, e.dtype().bits() * e.dtype().lanes() / 8);
  DataType uint16_type =
      DataType(uint8_type.code(), 16, uint8_type.bits() * uint8_type.lanes() / 16);
  DataType uint32_type =
      DataType(uint16_type.code(), 32, uint8_type.bits() * uint8_type.lanes() / 32);

  // Interpret input as vector of 8bit values
  PrimExpr input8 = reinterpret(uint8_type, e);
  // Popcount 8bit->8bit
  const CallNode* c0 = input8.as<CallNode>();
  ICHECK(c0 != nullptr);
  Array<PrimExpr> vcnt8_args;
  vcnt8_args.push_back(IntImm(DataType::UInt(32), ctpop_id));
  vcnt8_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt8_args.push_back(input8);
  PrimExpr vcnt8 = tir::Call(uint8_type, builtin_call_llvm_pure_intrin_, vcnt8_args);

  // Accumulation 8->16bit
  Array<PrimExpr> vcnt16_args;
  vcnt16_args.push_back(IntImm(DataType::UInt(32), vpaddlu_id));
  vcnt16_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt16_args.push_back(vcnt8);
  PrimExpr vcnt16 = tir::Call(uint16_type, builtin_call_llvm_pure_intrin_, vcnt16_args);
  if (call->dtype.bits() == 16) {
    return vcnt16;
  }

  // Accumulation 16->32bit
  Array<PrimExpr> vcnt32_args;
  vcnt32_args.push_back(IntImm(DataType::UInt(32), vpaddlu_id));
  vcnt32_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt32_args.push_back(vcnt16);
  PrimExpr vcnt32 = tir::Call(uint32_type, builtin_call_llvm_pure_intrin_, vcnt32_args);
  if (call->dtype.bits() == 32) {
    return vcnt32;
  }

  // Accumulation 32->64bit
  Array<PrimExpr> vcnt64_args;
  vcnt64_args.push_back(IntImm(DataType::UInt(32), vpaddlu_id));
  vcnt64_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt64_args.push_back(vcnt32);
  return tir::Call(call->dtype, builtin_call_llvm_pure_intrin_, vcnt64_args);
}

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_arm")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      *rv = static_cast<void*>(new CodeGenARM());
    });

#endif  // TVM_LLVM_VERSION

// ---  for aarch64 sve support 
#ifdef TVM_LLVM_VERSION >= 120
// -------------------------
// Utility functions to remove
void print_LLVM_type(llvm::Type* type) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  type->print(rso);
  LOG_INFO << rso.str() << std::endl;
}

void print_LLVM_val(llvm::Value* val) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  val->print(rso);
  LOG_INFO << rso.str() << std::endl;
}
// -------------------------

// AArch64 code generation
class CodeGenAArch64 final : public CodeGenARM {
 public:
  void InitTarget() {
    // set native vector bits.
    native_vector_bits_ = 16 * 8;
    CodeGenCPU::InitTarget();
  }
  llvm::Value* VisitExpr_(const BufferLoadNode* op);
  void VisitStmt_(const BufferStoreNode* op);
  void VisitStmt_(const ForNode* op);

  void BufferAccessHelper(
    Buffer buffer, Array<PrimExpr> indices, DataType value_dtype,
    std::function<llvm::Instruction*(TypedPointer buffer_ptr, int subelement_i, int alignment,
                                     bool is_volatile)>
        make_instruction);

 private:
  //
  // SVE LLVM intrinsics
  llvm::Value* sve_stride(int min_lanes);
  llvm::Value* sve_whilelt(llvm::Value* a, llvm::Value* b, int min_lanes);
  llvm::Value* sve_store(llvm::Value* ptr, llvm::Value* val, DataType t);
  llvm::Value* sve_load(llvm::Value* ptr, DataType t);
  void CreateSVEFor(llvm::Value* begin, llvm::Value* end, llvm::Value* stride, const Var& loop_var,
                    const Stmt& body, int min_lanes);

  // Predicate
  llvm::Value* mask_;
};

llvm::Value* CodeGenAArch64::sve_stride(int min_lanes) {
  llvm::Intrinsic::ID cnt_id;

  switch (min_lanes) {
    case 16:
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cntb");
      break;
    case 8:  // half
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cnth");
      break;
    case 4:  // float
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cntw");
      break;
    default:  // double
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cntd");
  }

  // All pattern
  int all_pattern = 31;

  llvm::Value* in_param = llvm::ConstantInt::get(t_int_, llvm::APInt(32, all_pattern));
  std::vector<llvm::Value*> arg_value{in_param};
  std::vector<llvm::Type*> arg_type{builder_->getInt32Ty()};
  llvm::Type* return_type = builder_->getInt64Ty();
  llvm::Function* func_cnt = GetIntrinsicDecl(cnt_id, return_type, arg_type);
  llvm::Value* vec_stride = builder_->CreateCall(func_cnt, arg_value);
  llvm::Value* vec_stride_int32 =
      builder_->CreateTruncOrBitCast(vec_stride, builder_->getInt32Ty());
  return vec_stride_int32;
}

llvm::Value* CodeGenAArch64::sve_whilelt(llvm::Value* a, llvm::Value* b, int min_lanes) {
  llvm::Intrinsic::ID whilelt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.whilelt");
  std::vector<llvm::Value*> arg_value{a, b};
  std::vector<llvm::Type*> arg_type{builder_->getInt32Ty(), builder_->getInt32Ty()};

  // Needs to be a vector type
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  llvm::Type* bool_type = llvm::Type::getIntNTy(*ctx, 1);
  llvm::Type* return_type = llvm::ScalableVectorType::get(bool_type, min_lanes);

  llvm::Function* func_whilelt = GetIntrinsicDecl(whilelt_id, return_type, arg_type);
  llvm::Value* whilelt = builder_->CreateCall(func_whilelt, arg_value);
  return whilelt;
}

llvm::Value* CodeGenAArch64::sve_store(llvm::Value* ptr, llvm::Value* val, DataType t) {
  // Get the intrinsic
  llvm::Intrinsic::ID st1_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.st1");
  std::vector<llvm::Value*> arg_value{val, mask_, ptr};

  // Get the pointer type
  llvm::PointerType* ptr_type = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
  ICHECK(ptr_type != nullptr);

  // Input types
  llvm::Type* mask_type = mask_->getType();
  llvm::Type* scalar_type = ptr_type->getElementType();
  llvm::Type* store_type = llvm::ScalableVectorType::get(scalar_type, t.lanes());
  std::vector<llvm::Type*> arg_type{store_type, mask_type, ptr_type};

  // Return type (void)
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  llvm::Type* return_type = llvm::Type::getVoidTy(*ctx);
  llvm::Function* func_st1 = GetIntrinsicDecl(st1_id, return_type, arg_type);

  // Create the call
  llvm::Value* st1 = builder_->CreateCall(func_st1, arg_value);
  return st1;
}

llvm::Value* CodeGenAArch64::sve_load(llvm::Value* ptr, DataType t) {
  llvm::Intrinsic::ID ld1_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.ld1");
  std::vector<llvm::Value*> arg_value{mask_, ptr};
  llvm::Type* ptr_type = ptr->getType();
  llvm::Type* mask_type = mask_->getType();

  std::vector<llvm::Type*> arg_type{mask_type, ptr_type};
  llvm::PointerType* ptype = llvm::dyn_cast<llvm::PointerType>(ptr_type);
  ICHECK(ptype != nullptr);

  llvm::Type* scalar_type = ptype->getElementType();
  llvm::Type* return_type = llvm::ScalableVectorType::get(scalar_type, t.lanes());
  llvm::Function* func_ld1 = GetIntrinsicDecl(ld1_id, return_type, arg_type);

  llvm::Value* ld1 = builder_->CreateCall(func_ld1, arg_value);
  return ld1;
}

void CodeGenAArch64::CreateSVEFor(llvm::Value* begin, llvm::Value* end, llvm::Value* stride,
                                  const Var& loop_var, const Stmt& body, int min_lanes) {
  using llvm::BasicBlock;
  BasicBlock* for_begin = builder_->GetInsertBlock();
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  BasicBlock* for_body = BasicBlock::Create(*ctx, "for_body", function_);
  BasicBlock* for_end = BasicBlock::Create(*ctx, "for_end", function_);

  // for_begin block
  builder_->SetInsertPoint(for_begin);
  llvm::Value* vec_stride = sve_stride(min_lanes);
  builder_->CreateBr(for_body);

  // for_body
  builder_->SetInsertPoint(for_body);
  llvm::PHINode* loop_value = builder_->CreatePHI(begin->getType(), 2);
  mask_ = sve_whilelt(loop_value, end, min_lanes);
  loop_value->addIncoming(begin, for_begin);
  ICHECK(!var_map_.count(loop_var.get()));
  var_map_[loop_var.get()] = loop_value;

  this->VisitStmt(body);
  var_map_.erase(loop_var.get());
  llvm::Value* loop_next = CreateAdd(loop_var.dtype(), loop_value, vec_stride);
  loop_value->addIncoming(loop_next, builder_->GetInsertBlock());
  builder_->CreateCondBr(CreateLT(loop_var.dtype(), loop_value, end), for_body, for_end,
                         md_very_likely_branch_);
  builder_->SetInsertPoint(for_end);
  function_->print(llvm::errs());
}

void CodeGenAArch64::BufferAccessHelper(
    Buffer buffer, Array<PrimExpr> indices, DataType value_dtype,
    std::function<llvm::Instruction*(TypedPointer buffer_ptr, int subelement_i, int alignment,
                                     bool is_volatile)>
        make_instruction) {
  DataType buffer_element_dtype = buffer->dtype;

  ICHECK_GE(indices.size(), 1)
      << "Buffer " << buffer->name << " is accessed with no indices.  "
      << "0-d scalar buffers are expected to be flattened to 1-d buffers prior to codegen.";

  // Only the last index is allowed to be multi-lane.  All earlier
  // indices must be scalar.  This only matters for subclasses of
  // CodeGenLLVM, because the default implementation of GetBufferPtr
  // requires 1-d indices.
  std::vector<llvm::Value*> earlier_index_values;
  for (size_t i = 0; i < indices.size() - 1; i++) {
    ICHECK_EQ(indices[i].dtype().lanes(), 1)
        << "Buffer " << buffer->name << " is accessed with a multi-lane index at position " << i
        << ".  Multi-lane indices are only supported as the last index.";
    earlier_index_values.push_back(MakeValue(indices[i]));
  }

  PrimExpr last_index = indices[indices.size() - 1];
  ICHECK_EQ(value_dtype.lanes(), last_index.dtype().lanes() * buffer_element_dtype.lanes());

  // Record index and elemtype in original form used for alias info
  PrimExpr last_index_origin = last_index;
  DataType buffer_element_dtype_origin = buffer_element_dtype;

  bool is_volatile = volatile_buf_.count(buffer->data.get());

  // If the buffer index is a contiguous ramp node, we only need to
  // access the first element, then cast to the value type.
  if (const RampNode* ramp_index = last_index.as<RampNode>()) {
    if (is_one(ramp_index->stride)) {
      last_index = ramp_index->base;
    }
  }

  // All TVM arrays are densely packed.  If the vectorized LLVM type
  // contains padding for alignment, we need to index based on the
  // size of the scalar type to avoid introducing that padding.
  if (last_index.dtype().lanes() == 1 && HasAlignmentPadding(buffer_element_dtype)) {
    last_index = buffer_element_dtype.lanes() * last_index;
    buffer_element_dtype = buffer_element_dtype.element_of();
  }

  int alignment;
  if (last_index.dtype().lanes() == 1) {
    // If we are accessing with a single index, then the vectorized
    // element being accessed may require more alignment than the
    // underlying data type.
    int native_bits;
    GetAlignment(value_dtype, buffer->data.get(), last_index, &alignment, &native_bits);
  } else {
    // Otherwise, alignment is based on the return value's scalar
    // type.
    ICHECK_GE(value_dtype.bits(), 8);
    alignment = value_dtype.bits() / 8;
  }

  llvm::Value* cached_vector_index = nullptr;
  for (int i = 0; i < last_index.dtype().lanes(); ++i) {
    llvm::Value* last_index_value;
    int subelement_i = i;
    if (const RampNode* ramp = last_index.as<RampNode>()) {
      PrimExpr offset = ramp->base + (ramp->stride * i);
      last_index_value = MakeValue(offset);
    } else if (last_index.dtype().lanes() > 1) {
      if (i == 0) {
        cached_vector_index = MakeValue(last_index);
      }
      last_index_value = builder_->CreateExtractElement(cached_vector_index, i);
    } else {
      last_index_value = MakeValue(last_index);
      subelement_i = -1;
    }

    std::vector<llvm::Value*> all_index_values = earlier_index_values;
    all_index_values.push_back(last_index_value);

    TypedPointer buffer_ptr =
        CreateBufferPtr(MakeValue(buffer->data), buffer_element_dtype, all_index_values,
                        value_dtype.with_lanes(value_dtype.lanes() / last_index.dtype().lanes()));
    auto instruction = make_instruction(buffer_ptr, subelement_i, alignment, is_volatile);
    AddAliasInfo(instruction, buffer->data.get(), last_index_origin, buffer_element_dtype_origin);
  }
}

llvm::Value* CodeGenAArch64::VisitExpr_(const BufferLoadNode* op) {
  DataType value_dtype = op->dtype;

  if (!value_dtype.is_scalable()) return CodeGenARM::VisitExpr_(op);

  std::vector<llvm::Value*> loads;

  auto make_load = [this, &loads](TypedPointer buffer_ptr, int /* subelement_i */, int alignment,
                                  bool is_volatile) {
#if TVM_LLVM_VERSION >= 110
    auto load = builder_->CreateAlignedLoad(buffer_ptr.type, buffer_ptr.addr,
                                            llvm::Align(alignment), is_volatile);
#elif TVM_LLVM_VERSION >= 80
    auto load =
        builder_->CreateAlignedLoad(buffer_ptr.type, buffer_ptr.addr, alignment, is_volatile);
#else
    auto load = builder_->CreateAlignedLoad(buffer_ptr.addr, alignment, is_volatile);
#endif

    loads.push_back(load);
    return load;
  };

  // Pass all indices into BufferAccessHelper.  In CodeGenLLVM,
  // non-flat indices will result in an error in CreateBufferPtr, but
  // a subclass may override CreateBufferPtr.
  // TODO(chenghao): use gather to address a load-with-stride-greater-than-1
  // ICHECK(is_one(ramp->stride));
  BufferAccessHelper(op->buffer, op->indices, value_dtype, make_load);
  
  if (loads.size() == 1) {
    return loads[0];
  } else {
    llvm::Value* ret = llvm::UndefValue::get(DTypeToLLVMType(value_dtype));
    for (size_t i = 0; i < loads.size(); i++) {
      ret = builder_->CreateInsertElement(ret, loads[i], ConstInt32(i));
    }
    return ret;
  }

  // // scalable vector load
  // const RampNode* ramp = op->index.as<RampNode>();
  // ICHECK(ramp);

  // int alignment, native_bits;
  // GetAlignment(t, op->buffer_var.get(), ramp->base, &alignment, &native_bits);
  // ICHECK_EQ(ramp->lanes, t.lanes());
  // llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, MakeValue(ramp->base));

  // llvm::Value* load = sve_load(ptr, t);
  // return load;
}

void CodeGenAArch64::VisitStmt_(const BufferStoreNode* op) {
//   ICHECK(is_one(op->predicate)) << op->predicate;
//   DataType t = op->value.dtype();
//   bool is_volatile = volatile_buf_.count(op->buffer_var.get());
//   llvm::Value* buffer = MakeValue(op->buffer_var);
//   llvm::Value* index = MakeValue(op->index);
//   llvm::Value* value = MakeValue(op->value);

//   if (t.lanes() == 1) {
//     int alignment, native_bits;
//     GetAlignment(t, op->buffer_var.get(), op->index, &alignment, &native_bits);
//     llvm::Value* ptr = CreateBufferPtr(t, buffer, index);
// #if TVM_LLVM_VERSION >= 110
//     llvm::StoreInst* store =
//         builder_->CreateAlignedStore(value, ptr, llvm::Align(alignment), is_volatile);
// #else
//     llvm::StoreInst* store = builder_->CreateAlignedStore(value, ptr, alignment, is_volatile);
// #endif
//     AddAliasInfo(store, op->buffer_var.get(), op->index);
//     return;
//   } else {
//     // vector store
//     unsigned addrspace = llvm::dyn_cast<llvm::PointerType>(buffer->getType())->getAddressSpace();
//     if (const RampNode* ramp = op->index.as<RampNode>()) {
//       if (is_one(ramp->stride)) {
//         int alignment, native_bits;
//         GetAlignment(t, op->buffer_var.get(), ramp->base, &alignment, &native_bits);
//         ICHECK_EQ(ramp->lanes, t.lanes());
//         llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, MakeValue(ramp->base));
//         if (!t.is_scalable()) {
//           ptr = builder_->CreatePointerCast(ptr, DTypeToLLVMType(t)->getPointerTo(addrspace));
//         }
// #if TVM_LLVM_VERSION >= 110
//         if (t.is_scalable()) {
//           sve_store(ptr, value, t);
//           return;
//         }
//         llvm::StoreInst* store =
//             builder_->CreateAlignedStore(value, ptr, llvm::Align(alignment), is_volatile);
// #else
//         llvm::StoreInst* store = builder_->CreateAlignedStore(value, ptr, alignment, is_volatile);
// #endif
//         AddAliasInfo(store, op->buffer_var.get(), op->index);
//         return;
//       }
//     }
//   }
//   ICHECK_GE(t.bits(), 8);
//   // scalarized store.
//   int basic_align = t.bits() / 8;
//   auto f = [&](int i, llvm::Value* index) {
//     llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, index);
// #if TVM_LLVM_VERSION >= 110
//     llvm::StoreInst* store = builder_->CreateAlignedStore(
//         builder_->CreateExtractElement(value, i), ptr, llvm::Align(basic_align), is_volatile);
// #else
//     llvm::StoreInst* store = builder_->CreateAlignedStore(builder_->CreateExtractElement(value, i),
//                                                           ptr, basic_align, is_volatile);
// #endif
//     AddAliasInfo(store, op->buffer_var.get(), PrimExpr());
//   };
//   this->Scalarize(op->index, f);
}

void CodeGenAArch64::VisitStmt_(const ForNode* op) {
  ICHECK(is_zero(op->min));
  analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  if (op->is_vla) {
    CreateSVEFor(MakeValue(op->min), 
                 MakeValue(op->extent),
                 llvm::ConstantInt::getSigned(GetLLVMType(op->extent), 1), 
                 op->loop_var, 
                 op->body,
                 op->stride);
  } else {
    CodeGenARM::VisitStmt_(op);
  }
}

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_aarch64")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      CodeGenLLVM* cg = new CodeGenAArch64();
      *rv = static_cast<void*>(cg);
    });

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
