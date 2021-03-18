use super::{ir_expr::IrExprTrait, types::HirType, HirMode};
use crate::hir::modes::memory_bounds::MemBoundTrait;
use crate::hir::modes::types::TypedTrait;
use crate::hir::{modes::DepAnaMode, modes::*, Hir};
use crate::{hir::modes::dependencies::DepAnaTrait, hir::modes::ordering::OrderedTrait};

// MODE IMPLS
impl HirMode for IrExprMode {}
impl HirMode for DepAnaMode {}
impl HirMode for TypedMode {}
impl HirMode for OrderedMode {}
impl HirMode for MemBoundMode {}
impl HirMode for CompleteMode {}

// WRAPPERS

pub(crate) trait TypedWrapper {
    type InnerT: TypedTrait;
    fn inner_typed(&self) -> &Self::InnerT;
}

pub(crate) trait OrderedWrapper {
    type InnerO: OrderedTrait;
    fn inner_order(&self) -> &Self::InnerO;
}
pub(crate) trait DepAnaWrapper {
    type InnerD: DepAnaTrait;
    fn inner_dep(&self) -> &Self::InnerD;
}

pub(crate) trait MemoryWrapper {
    type InnerM: MemBoundTrait;
    fn inner_memory(&self) -> &Self::InnerM;
}

pub trait IrExprWrapper {
    type InnerE: IrExprTrait;
    fn inner_expr(&self) -> &Self::InnerE;
}

// WRAPPER HIR IMPLEMENTATIONS
impl<M> DepAnaWrapper for Hir<M>
where
    M: DepAnaTrait + HirMode,
{
    type InnerD = M;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.mode
    }
}

impl<M> IrExprWrapper for Hir<M>
where
    M: IrExprTrait + HirMode,
{
    type InnerE = M;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.mode
    }
}

impl<M> OrderedWrapper for Hir<M>
where
    M: OrderedTrait + HirMode,
{
    type InnerO = M;
    fn inner_order(&self) -> &Self::InnerO {
        &self.mode
    }
}

impl<M> TypedWrapper for Hir<M>
where
    M: TypedTrait + HirMode,
{
    type InnerT = M;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.mode
    }
}

impl<M> MemoryWrapper for Hir<M>
where
    M: MemBoundTrait + HirMode,
{
    type InnerM = M;
    fn inner_memory(&self) -> &Self::InnerM {
        &self.mode
    }
}

// BLANKET IMPLEMENTATIONS
impl<A: TypedWrapper<InnerT = T>, T: TypedTrait + 'static> TypedTrait for A {
    fn stream_type(&self, sr: SRef) -> HirType {
        self.inner_typed().stream_type(sr)
    }
    fn is_periodic(&self, sr: SRef) -> bool {
        self.inner_typed().is_periodic(sr)
    }
    fn is_event(&self, sr: SRef) -> bool {
        self.inner_typed().is_event(sr)
    }
    fn expr_type(&self, eid: ExprId) -> HirType {
        self.inner_typed().expr_type(eid)
    }
}
impl<A: OrderedWrapper<InnerO = T>, T: OrderedTrait + 'static> OrderedTrait for A {
    fn stream_layers(&self, sr: SRef) -> StreamLayers {
        self.inner_order().stream_layers(sr)
    }
}
impl<A: MemoryWrapper<InnerM = T>, T: MemBoundTrait + 'static> MemBoundTrait for A {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound {
        self.inner_memory().memory_bound(sr)
    }
}

impl<A: IrExprWrapper<InnerE = T>, T: IrExprTrait + 'static> IrExprTrait for A {
    fn window_refs(&self) -> Vec<WRef> {
        self.inner_expr().window_refs()
    }
    fn single_window(&self, window: WRef) -> Either<SlidingWindow, DiscreteWindow> {
        self.inner_expr().single_window(window)
    }

    fn expression(&self, id: ExprId) -> &Expression {
        self.inner_expr().expression(id)
    }
    fn func_declaration(&self, func_name: &str) -> &FuncDecl {
        self.inner_expr().func_declaration(func_name)
    }
}
impl<A: DepAnaWrapper<InnerD = T>, T: DepAnaTrait + 'static> DepAnaTrait for A {
    fn direct_accesses(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().direct_accesses(who)
    }

    fn transitive_accesses(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().transitive_accesses(who)
    }

    fn direct_accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().direct_accessed_by(who)
    }

    fn transitive_accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().transitive_accessed_by(who)
    }

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.inner_dep().aggregated_by(who)
    }

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.inner_dep().aggregates(who)
    }

    fn graph(&self) -> &DependencyGraph {
        self.inner_dep().graph()
    }
}

// EXTENSION IMPLEMENTATION

impl IrExprWrapper for DepAnaMode {
    type InnerE = IrExprMode;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

// All Below Typed

impl IrExprWrapper for TypedMode {
    type InnerE = IrExprMode;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DepAnaWrapper for TypedMode {
    type InnerD = DepAnaMode;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

// All Below Ordered

impl IrExprWrapper for OrderedMode {
    type InnerE = IrExprMode;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DepAnaWrapper for OrderedMode {
    type InnerD = DepAnaMode;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl TypedWrapper for OrderedMode {
    type InnerT = TypedMode;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

// All below Membound

impl IrExprWrapper for MemBoundMode {
    type InnerE = IrExprMode;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DepAnaWrapper for MemBoundMode {
    type InnerD = DepAnaMode;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl TypedWrapper for MemBoundMode {
    type InnerT = TypedMode;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

impl OrderedWrapper for MemBoundMode {
    type InnerO = OrderedMode;
    fn inner_order(&self) -> &Self::InnerO {
        &self.layers
    }
}

// All below Complete

impl IrExprWrapper for CompleteMode {
    type InnerE = IrExprMode;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DepAnaWrapper for CompleteMode {
    type InnerD = DepAnaMode;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl TypedWrapper for CompleteMode {
    type InnerT = TypedMode;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

impl OrderedWrapper for CompleteMode {
    type InnerO = OrderedMode;
    fn inner_order(&self) -> &Self::InnerO {
        &self.layers
    }
}

impl MemoryWrapper for CompleteMode {
    type InnerM = Memory;
    fn inner_memory(&self) -> &Self::InnerM {
        &self.memory
    }
}
