pub(crate) trait HirMode {}

pub(crate) mod blanket;
pub(crate) mod complete;
pub(crate) mod dependencies;
pub(crate) mod ir_expr;
pub(crate) mod raw;
pub(crate) mod types;

use std::collections::HashMap;

use crate::{
    ast, common_ir::MemorizationBound, common_ir::StreamReference as SRef, common_ir::WindowReference as WRef,
    hir::expression::Expression, hir::Hir,
};

use self::dependencies::DependencyErr;

use super::Window;

pub(crate) struct Raw {
    constants: Vec<ast::Constant>,
    expressions: HashMap<SRef, ast::Expression>,
}
impl HirMode for Raw {}

pub(crate) struct IrExpression {
    expressions: HashMap<SRef, Expression>,
    windows: Vec<Window>,
}
impl HirMode for IrExpression {}

impl Hir<IrExpression> {
    pub(crate) fn build_dependency_graph(self) -> Result<Hir<Dependencies>, DependencyErr> {
        let dep = Dependencies::analyze(&self)?;
        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode: dep,
        })
    }
}

struct DependencyGraph {
    accesses: HashMap<SRef, Vec<SRef>>,
    accessed_by: HashMap<SRef, Vec<SRef>>,
    aggregated_by: HashMap<SRef, Vec<(SRef, WRef)>>,
    aggregates: HashMap<SRef, Vec<(SRef, WRef)>>,
}
pub(crate) struct Dependencies {
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
}
impl HirMode for Dependencies {}

impl Hir<Dependencies> {
    pub(crate) fn type_check(self) -> Hir<Typed> {
        unimplemented!()
    }
}

pub(crate) struct HirType {} // TBD
pub(crate) struct Typed {
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
    stream_tt: HashMap<SRef, HirType>,
    expr_tt: HashMap<SRef, HirType>, // consider merging the tts.
}
impl HirMode for Typed {}

impl Hir<Typed> {
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

pub(crate) trait MemoryAnalyzed {
    fn memory(&self, sr: SRef) -> MemorizationBound;
}

pub(crate) struct Complete {
    memory: HashMap<SRef, MemorizationBound>,
    dependencies: Dependencies,
    ir_expr: IrExpression,
    types: Typed,
}
impl HirMode for Complete {}

pub(crate) trait AstExpr {
    fn expr(&self, sr: SRef) -> ast::Expression;
}
