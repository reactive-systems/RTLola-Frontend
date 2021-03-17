use super::{ir_expr::WithIrExpr, types::HirType, HirMode};
use crate::hir::modes::memory_bounds::MemoryAnalyzed;
use crate::hir::modes::types::TypeChecked;
use crate::hir::{modes::DependencyAnalyzed, modes::*, Hir};
use crate::{hir::modes::dependencies::WithDependencies, hir::modes::ordering::EvaluationOrderBuilt};

// MODE IMPLS
impl HirMode for IrExpression {}
impl HirMode for DependencyAnalyzed {}
impl HirMode for Typed {}
impl HirMode for Ordered {}
impl HirMode for MemBound {}
impl HirMode for Complete {}

// WRAPPERS

pub(crate) trait TypedWrapper {
    type InnerT: TypeChecked;
    fn inner_typed(&self) -> &Self::InnerT;
}

pub(crate) trait OrderedWrapper {
    type InnerO: EvaluationOrderBuilt;
    fn inner_order(&self) -> &Self::InnerO;
}
pub(crate) trait DependenciesWrapper {
    type InnerD: WithDependencies;
    fn inner_dep(&self) -> &Self::InnerD;
}

pub(crate) trait MemoryWrapper {
    type InnerM: MemoryAnalyzed;
    fn inner_memory(&self) -> &Self::InnerM;
}

pub trait IrExprWrapper {
    type InnerE: WithIrExpr;
    fn inner_expr(&self) -> &Self::InnerE;
}

// WRAPPER IMPLEMENTATIONS
impl TypedWrapper for Typed {
    type InnerT = TypeTable;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.tts
    }
}

impl OrderedWrapper for Ordered {
    type InnerO = EvaluationOrder;
    fn inner_order(&self) -> &Self::InnerO {
        &self.layers
    }
}
impl MemoryWrapper for MemBound {
    type InnerM = Memory;

    fn inner_memory(&self) -> &Self::InnerM {
        &self.memory
    }
}
impl DependenciesWrapper for DependencyAnalyzed {
    type InnerD = Dependencies;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl IrExprWrapper for IrExpression {
    type InnerE = IrExprRes;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr_res
    }
}

// WRAPPER HIR IMPLEMENTATIONS
impl<M> DependenciesWrapper for Hir<M>
where
    M: WithDependencies + HirMode,
{
    type InnerD = M;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.mode
    }
}

impl<M> IrExprWrapper for Hir<M>
where
    M: WithIrExpr + HirMode,
{
    type InnerE = M;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.mode
    }
}

impl<M> OrderedWrapper for Hir<M>
where
    M: EvaluationOrderBuilt + HirMode,
{
    type InnerO = M;
    fn inner_order(&self) -> &Self::InnerO {
        &self.mode
    }
}

impl<M> TypedWrapper for Hir<M>
where
    M: TypeChecked + HirMode,
{
    type InnerT = M;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.mode
    }
}

impl<M> MemoryWrapper for Hir<M>
where
    M: MemoryAnalyzed + HirMode,
{
    type InnerM = M;
    fn inner_memory(&self) -> &Self::InnerM {
        &self.mode
    }
}

// BLANKET IMPLEMENTATIONS
impl<A: TypedWrapper<InnerT = T>, T: TypeChecked + 'static> TypeChecked for A {
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
impl<A: OrderedWrapper<InnerO = T>, T: EvaluationOrderBuilt + 'static> EvaluationOrderBuilt for A {
    fn stream_layers(&self, sr: SRef) -> StreamLayers {
        self.inner_order().stream_layers(sr)
    }
}
impl<A: MemoryWrapper<InnerM = T>, T: MemoryAnalyzed + 'static> MemoryAnalyzed for A {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound {
        self.inner_memory().memory_bound(sr)
    }
}

impl<A: IrExprWrapper<InnerE = T>, T: WithIrExpr + 'static> WithIrExpr for A {
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
impl<A: DependenciesWrapper<InnerD = T>, T: WithDependencies + 'static> WithDependencies for A {
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

impl IrExprWrapper for DependencyAnalyzed {
    type InnerE = IrExprRes;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

// All Below Typed

impl IrExprWrapper for Typed {
    type InnerE = IrExprRes;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DependenciesWrapper for Typed {
    type InnerD = Dependencies;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

// All Below Ordered

impl IrExprWrapper for Ordered {
    type InnerE = IrExprRes;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DependenciesWrapper for Ordered {
    type InnerD = Dependencies;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl TypedWrapper for Ordered {
    type InnerT = TypeTable;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

// All below Membound

impl IrExprWrapper for MemBound {
    type InnerE = IrExprRes;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DependenciesWrapper for MemBound {
    type InnerD = Dependencies;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl TypedWrapper for MemBound {
    type InnerT = TypeTable;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

impl OrderedWrapper for MemBound {
    type InnerO = EvaluationOrder;
    fn inner_order(&self) -> &Self::InnerO {
        &self.layers
    }
}

// All below Complete

impl IrExprWrapper for Complete {
    type InnerE = IrExprRes;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DependenciesWrapper for Complete {
    type InnerD = Dependencies;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl TypedWrapper for Complete {
    type InnerT = TypeTable;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

impl OrderedWrapper for Complete {
    type InnerO = EvaluationOrder;
    fn inner_order(&self) -> &Self::InnerO {
        &self.layers
    }
}

impl MemoryWrapper for Complete {
    type InnerM = Memory;
    fn inner_memory(&self) -> &Self::InnerM {
        &self.memory
    }
}
