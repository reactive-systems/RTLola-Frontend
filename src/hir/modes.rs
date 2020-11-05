pub(crate) trait HirMode {}

pub(crate) mod blanket;
pub(crate) mod complete;
pub(crate) mod dependencies;
pub(crate) mod dg_functionality;
pub(crate) mod ir_expr;
pub(crate) mod memory_bounds;
pub(crate) mod ordering;
pub(crate) mod raw;
pub(crate) mod types;

use crate::hir::function_lookup::FuncDecl;
use crate::hir::SlidingWindow;
use std::collections::HashMap;

use crate::{
    ast, ast::Ast, common_ir::MemorizationBound, common_ir::StreamLayers, common_ir::StreamReference as SRef,
    common_ir::WindowReference as WRef, hir::expression::Expression, hir::modes::types::HirType, hir::ExprId, hir::Hir,
    reporting::Handler, FrontendConfig,
};

use self::dependencies::DependencyErr;
use petgraph::Graph;

pub(crate) struct Raw {}
impl HirMode for Raw {}

impl Hir<Raw> {
    #[allow(unused_variables)]
    pub(crate) fn transform_expressions(self, handler: &Handler, config: &FrontendConfig) -> Hir<IrExpression> {
        //Hir::<IrExpression>::transform_expressions(self, handler, config)
        todo!()
    }
}

pub(crate) type ExpressionLookUps = HashMap<ExprId, Expression>;
pub(crate) type WindowLookUps = HashMap<ExprId, SlidingWindow>;
pub(crate) type FunctionLookUps = HashMap<String, FuncDecl>;

pub(crate) struct IrExpression {
    exprid_to_expr: ExpressionLookUps,
    windows: WindowLookUps,
    func_table: FunctionLookUps,
}
impl HirMode for IrExpression {}

impl Hir<IrExpression> {
    pub(crate) fn from_ast(ast: Ast, handler: &Handler, config: &FrontendConfig) -> Self {
        Hir::<IrExpression>::transform_expressions(ast, handler, config)
    }

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
    Offset(i32),
    Aggr(WRef),
    Hold,
    Spawn(Box<EdgeWeight>),
    Filter(Box<EdgeWeight>),
    Close(Box<EdgeWeight>),
}

pub(crate) type Streamdependencies = HashMap<SRef, Vec<SRef>>;
pub(crate) type Windowdependencies = HashMap<SRef, Vec<(SRef, WRef)>>;
pub(crate) type Graphrepresentation = Graph<SRef, EdgeWeight>;

struct DependencyGraph {
    accesses: Streamdependencies,
    accessed_by: Streamdependencies,
    aggregated_by: Windowdependencies,
    aggregates: Windowdependencies,
    graph: Graphrepresentation,
}
pub(crate) struct Dependencies {
    ir_expr: IrExpression,
    dg: DependencyGraph,
}
impl HirMode for Dependencies {}

impl Hir<Dependencies> {
    pub(crate) fn type_check(self) -> Hir<Typed> {
        unimplemented!()
    }
}

pub(crate) type StreamTypeTable = HashMap<SRef, HirType>;
pub(crate) type ExpressionTypeTable = HashMap<SRef, HirType>; // -> why is expressionid not the key for this map

pub(crate) struct TypeTables {
    stream_tt: StreamTypeTable,
    expr_tt: ExpressionTypeTable, // consider merging the tts.
}

pub(crate) struct Typed {
    ir_expr: IrExpression,
    dg: DependencyGraph,
    tts: TypeTables,
}
impl HirMode for Typed {}

impl Hir<Typed> {
    pub(crate) fn build_evaluation_order(self) -> Hir<Ordered> {
        unimplemented!()
    }
}

pub(crate) type LayerRepresentation = HashMap<SRef, StreamLayers>;

pub(crate) struct EvaluationOrder {
    event_layers: LayerRepresentation,
    periodic_layers: LayerRepresentation,
}
pub(crate) struct Ordered {
    ir_expr: IrExpression,
    dependencies: Dependencies,
    types: Typed,
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
    ir_expr: IrExpression,
    dependencies: Dependencies,
    types: Typed,
    layers: EvaluationOrder,
    memory: Memory,
}
impl HirMode for MemBound {}

impl Hir<MemBound> {
    pub(crate) fn finalize(self) -> Hir<Complete> {
        unimplemented!()
    }
}

pub(crate) struct Complete {
    ir_expr: IrExpression,
    dependencies: Dependencies,
    types: Typed,
    layers: EvaluationOrder,
    memory: Memory,
}
impl HirMode for Complete {}

pub(crate) trait AstExpr {
    fn expr(&self, sr: SRef) -> ast::Expression;
}
