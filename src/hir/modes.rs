pub(crate) trait HirMode {}

pub(crate) mod blanket;
pub(crate) mod common_functionality;
pub(crate) mod complete;
pub(crate) mod dependencies;
pub(crate) mod ir_expr;
pub(crate) mod memory_bounds;
pub(crate) mod ordering;
pub(crate) mod raw;
pub(crate) mod types;

use std::collections::HashMap;

use crate::{
    ast, common_ir::MemorizationBound, common_ir::StreamLayers, common_ir::StreamReference as SRef,
    common_ir::WindowReference as WRef, hir::expression::Expression, hir::Hir, reporting::Handler, FrontendConfig,
};

use self::dependencies::DependencyErr;
use petgraph::Graph;

use super::Window;

pub(crate) struct Raw {
    constants: Vec<ast::Constant>,
    expressions: HashMap<SRef, ast::Expression>,
}
impl HirMode for Raw {}

impl Hir<Raw> {
    pub(crate) fn transform_expressions(self, handler: &Handler, config: &FrontendConfig) -> Hir<IrExpression> {
        Hir::<IrExpression>::transform_expressions(self, handler, config)
    }
}

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

#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub(crate) enum EdgeWeight {
    Infinite,
    Offset(i32),
    Aggr(WRef),
    Hold,
    Spawn(Box<EdgeWeight>),
    Filter(Box<EdgeWeight>),
    Close(Box<EdgeWeight>),
}

struct DependencyGraph {
    accesses: HashMap<SRef, Vec<SRef>>,
    accessed_by: HashMap<SRef, Vec<SRef>>,
    aggregated_by: HashMap<SRef, Vec<(SRef, WRef)>>,
    aggregates: HashMap<SRef, Vec<(SRef, WRef)>>,
    graph: Graph<SRef, EdgeWeight>,
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
    pub(crate) fn build_evaluation_order(self) -> Hir<Ordered> {
        unimplemented!()
    }
}

pub(crate) struct EvaluationOrder {
    event_layers: HashMap<SRef, StreamLayers>,
    periodic_layers: HashMap<SRef, StreamLayers>,
}
pub(crate) struct Ordered {
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
    stream_tt: HashMap<SRef, HirType>,
    expr_tt: HashMap<SRef, HirType>,
    layers: EvaluationOrder,
}
impl HirMode for Ordered {}

impl Hir<Ordered> {
    pub(crate) fn compute_memory_bounds(self) -> Hir<MemBound> {
        unimplemented!()
    }
}
pub(crate) struct Memory {
    memory_bound_per_stram: HashMap<SRef, MemorizationBound>,
}
pub(crate) struct MemBound {
    memory: Memory,
    expressions: HashMap<SRef, Expression>,
    dg: DependencyGraph,
    stream_tt: HashMap<SRef, HirType>,
    expr_tt: HashMap<SRef, HirType>, // consider merging the tts.
    layers: EvaluationOrder,
}
impl HirMode for MemBound {}

impl Hir<MemBound> {
    pub(crate) fn finalize(self) -> Hir<Complete> {
        unimplemented!()
    }
}

pub(crate) struct Complete {
    memory: Memory,
    dependencies: Dependencies,
    ir_expr: IrExpression,
    types: Typed,
    layers: EvaluationOrder,
}
impl HirMode for Complete {}

pub(crate) trait AstExpr {
    fn expr(&self, sr: SRef) -> ast::Expression;
}
