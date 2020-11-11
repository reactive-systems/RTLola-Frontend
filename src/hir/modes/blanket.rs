use super::{ir_expr::WithIrExpr, HirMode};
use crate::hir::modes::ir_expr::IrExprWrapper;
use crate::hir::modes::memory_bounds::MemoryAnalyzed;
use crate::hir::modes::types::TypeChecked;
use crate::hir::modes::types::TypedWrapper;
use crate::hir::{
    modes::dependencies::DependenciesWrapper, modes::memory_bounds::MemoryWrapper, modes::ordering::OrderedWrapper, Hir,
};
use crate::{hir::modes::dependencies::WithDependencies, hir::modes::ordering::EvaluationOrderBuilt};

impl<M> IrExprWrapper for Hir<M>
where
    M: WithIrExpr + HirMode,
{
    type InnerE = M;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.mode
    }
}

impl<M> DependenciesWrapper for Hir<M>
where
    M: WithDependencies + HirMode,
{
    type InnerD = M;
    fn inner_dep(&self) -> &Self::InnerD {
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
