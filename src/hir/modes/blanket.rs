use super::{ir_expr::WithIrExpr, HirMode};
use crate::hir::modes::ir_expr::IrExprWrapper;
use crate::hir::modes::memory_bounds::MemoryAnalyzed;
use crate::hir::modes::types::TypeChecked;
use crate::hir::modes::types::TypedWrapper;
use crate::hir::{
    modes::dependencies::DependenciesWrapper, modes::memory_bounds::MemoryWrapper, modes::ordering::OrderedWrapper,
    modes::DependencyAnalyzed, modes::*, Hir,
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

// All Below IrExpression - impl IrExprWrapper

// All Below Dep

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
        &self.dg
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
    type InnerT = TypeTables;
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
    type InnerT = TypeTables;
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
    type InnerT = TypeTables;
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
