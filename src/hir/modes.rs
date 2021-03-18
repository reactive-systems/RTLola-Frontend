pub trait HirMode {}

pub(crate) mod blanket;
pub(crate) mod complete;
pub(crate) mod dependencies;
pub(crate) mod dg_functionality;
pub mod ir_expr;
pub(crate) mod memory_bounds;
pub(crate) mod ordering;
pub(crate) mod raw;
pub(crate) mod types;

use crate::hir::function_lookup::FuncDecl;
use crate::hir::{DiscreteWindow, SlidingWindow};
use std::collections::HashMap;

use crate::{
    ast, ast::Ast, common_ir::MemorizationBound, common_ir::StreamLayers, common_ir::StreamReference as SRef,
    common_ir::WindowReference as WRef, hir::expression::Expression, hir::ExprId, hir::Hir, reporting::Handler,
    tyc::rtltc::TypeTable, FrontendConfig,
};

use self::dependencies::DependencyErr;
use itertools::Either;
use petgraph::stable_graph::StableGraph;

pub(crate) struct Raw {}
impl HirMode for Raw {}

impl Hir<Raw> {
    #[allow(unused_variables)]
    pub(crate) fn transform_expressions(self, handler: &Handler, config: &FrontendConfig) -> Hir<IrExprMode> {
        //Hir::<IrExpression>::transform_expressions(self, handler, config)
        todo!()
    }
}

pub type ExpressionLookUps = HashMap<ExprId, Expression>;
pub type WindowLookUps = HashMap<WRef, Either<SlidingWindow, DiscreteWindow>>;
pub type FunctionLookUps = HashMap<String, FuncDecl>;

#[derive(Clone, Debug)]
pub struct IrExprRes {
    exprid_to_expr: ExpressionLookUps,
    windows: WindowLookUps,
    func_table: FunctionLookUps,
}

#[derive(Clone, Debug)]
pub struct IrExprMode {
    ir_expr_res: IrExprRes,
}

impl Hir<IrExprMode> {
    pub fn from_ast(ast: Ast, handler: &Handler, config: &FrontendConfig) -> Self {
        Hir::<IrExprMode>::transform_expressions(ast, handler, config)
    }

    pub(crate) fn build_dependency_graph(self) -> Result<Hir<DepAnaMode>, DependencyErr> {
        let dependencies = Dependencies::analyze(&self)?;
        let mode = DepAnaMode { ir_expr: self.mode, dependencies };
        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
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
pub(crate) type DependencyGraph = StableGraph<SRef, EdgeWeight>;

#[derive(Debug, Clone)]
pub(crate) struct Dependencies {
    direct_accesses: Streamdependencies,
    transitive_accesses: Streamdependencies,
    direct_accessed_by: Streamdependencies,
    transitive_accessed_by: Streamdependencies,
    aggregated_by: Windowdependencies,
    aggregates: Windowdependencies,
    graph: DependencyGraph,
}
#[derive(Debug, Clone)]
pub(crate) struct DepAnaMode {
    ir_expr: IrExprMode,
    dependencies: Dependencies,
}

impl Hir<DepAnaMode> {
    pub(crate) fn type_check(self, handler: &Handler) -> Result<Hir<TypedMode>, String> {
        let tts = crate::tyc::type_check(&self, handler)?;

        let mode = TypedMode { ir_expr: self.mode.ir_expr.clone(), dependencies: self.mode, tts };
        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
        })
    }
}
/*
pub(crate) type StreamTypeTable = HashMap<SRef, HirType>;
pub(crate) type ExpressionTypeTable = HashMap<SRef, HirType>; // -> why is expressionid not the key for this map

#[derive(Debug, Clone)]
pub(crate) struct TypeTables {
    stream_tt: StreamTypeTable,
    expr_tt: ExpressionTypeTable, // consider merging the tts.
}
*/
#[derive(Debug, Clone)]
pub(crate) struct TypedMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    tts: TypeTable,
}

impl Hir<TypedMode> {
    pub(crate) fn build_evaluation_order(self) -> Hir<OrderedMode> {
        let order = EvaluationOrder::analyze(&self);

        let old_mode = self.mode.clone();
        let mode = OrderedMode {
            ir_expr: self.mode.ir_expr,
            dependencies: self.mode.dependencies,
            types: old_mode,
            layers: order,
        };

        Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
        }
    }
}

pub(crate) type LayerRepresentation = HashMap<SRef, StreamLayers>;

#[derive(Debug, Clone)]
pub(crate) struct EvaluationOrder {
    event_layers: LayerRepresentation,
    periodic_layers: LayerRepresentation,
}
#[derive(Debug, Clone)]
pub(crate) struct OrderedMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    types: TypedMode,
    layers: EvaluationOrder,
}

impl Hir<OrderedMode> {
    pub(crate) fn compute_memory_bounds(self) -> Hir<MemBoundMode> {
        //TODO: forward config argument
        let memory = Memory::analyze(&self, false);

        let old_mode = self.mode.clone();
        let mode = MemBoundMode {
            ir_expr: self.mode.ir_expr,
            dependencies: self.mode.dependencies,
            types: self.mode.types,
            layers: old_mode,
            memory,
        };

        Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Memory {
    memory_bound_per_stream: HashMap<SRef, MemorizationBound>,
}

#[derive(Debug, Clone)]
pub(crate) struct MemBoundMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    types: TypedMode,
    layers: OrderedMode,
    memory: Memory,
}

impl Hir<MemBoundMode> {
    pub(crate) fn finalize(self) -> Hir<CompleteMode> {
        let mode = CompleteMode {
            ir_expr: self.mode.ir_expr,
            dependencies: self.mode.dependencies,
            types: self.mode.types,
            layers: self.mode.layers,
            memory: self.mode.memory,
        };

        Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CompleteMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    types: TypedMode,
    layers: OrderedMode,
    memory: Memory,
}

pub(crate) trait AstExpr {
    fn expr(&self, sr: SRef) -> ast::Expression;
}
