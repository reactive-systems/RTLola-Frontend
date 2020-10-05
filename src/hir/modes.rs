pub(crate) trait HirMode {}

use std::collections::HashMap;

use crate::{
    ast, common_ir::MemorizationBound, common_ir::StreamReference as SRef, hir::expression::ExprId,
    hir::expression::Expression, hir::Hir,
};

pub(crate) struct Raw {
    constants: Vec<ast::Constant>,
    expressions: HashMap<SRef, ast::Expression>,
}
impl HirMode for Raw {}

pub(crate) struct IrExpression {
    expressions: HashMap<SRef, Expression>,
}
impl HirMode for IrExpression {}

impl Hir<IrExpression> {
    pub(crate) fn build_dependency_graph(self) -> Hir<DepAna> {
        unimplemented!()
    }
}

struct DependencyGraph;
pub(crate) struct DepAna {
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
}
impl HirMode for DepAna {}

impl Hir<DepAna> {
    pub(crate) fn type_check(self) -> Hir<TypeChecked> {
        unimplemented!()
    }
}

pub(crate) struct HirType {} // TBD
pub(crate) struct TypeChecked {
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
    stream_tt: HashMap<SRef, HirType>,
    expr_tt: HashMap<SRef, HirType>, // consider merging the tts.
}
impl HirMode for TypeChecked {}

impl Hir<TypeChecked> {
    pub(crate) fn stream_type(&self, _sr: SRef) -> HirType {
        unimplemented!()
    }
    pub(crate) fn expr_type(&self, _eid: ExprId) -> HirType {
        unimplemented!()
    }
    pub(crate) fn compute_memory_bounds(self) -> Hir<MemBound> {
        unimplemented!()
    }
}

pub(crate) struct MemBound {
    memory: HashMap<SRef, MemorizationBound>,
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
    stream_tt: HashMap<SRef, HirType>,
    expr_tt: HashMap<SRef, HirType>, // consider merging the tts.
}
impl HirMode for MemBound {}

impl Hir<MemBound> {
    pub(crate) fn finalize(self) -> Hir<Complete> {
        unimplemented!()
    }
}

pub(crate) struct Complete {
    memory: HashMap<SRef, MemorizationBound>,
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
    stream_tt: HashMap<SRef, HirType>,
    expr_tt: HashMap<SRef, HirType>, // consider merging the tts.
}
impl HirMode for Complete {}

pub(crate) mod complete;
pub(crate) mod raw;
