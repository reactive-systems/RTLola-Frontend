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
use itertools::Itertools;
use std::collections::HashMap;

use crate::{
    common_ir::MemorizationBound, common_ir::StreamLayers, common_ir::StreamReference as SRef,
    common_ir::WindowReference as WRef, hir::expression::Expression, hir::ExprId, hir::Hir, reporting::Handler,
    tyc::rtltc::TypeTable, FrontendConfig,
};

use self::{
    dependencies::{DependencyGraph, Streamdependencies, Windowdependencies},
    memory_bounds::LayerRepresentation,
    types::HirType,
};
use itertools::Either;

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

#[derive(Clone, Debug, HirMode)]
pub struct IrExprMode {
    ir_expr_res: IrExprRes,
}

#[mode_functionality]
pub trait IrExprTrait {
    fn window_refs(&self) -> Vec<WRef>;
    fn all_windows(&self) -> (Vec<SlidingWindow>, Vec<DiscreteWindow>) {
        self.window_refs().into_iter().partition_map(|w| self.single_window(w))
    }
    fn sliding_windows(&self) -> Vec<SlidingWindow> {
        self.all_windows().0
    }
    fn discrete_windows(&self) -> Vec<DiscreteWindow> {
        self.all_windows().1
    }
    fn single_window(&self, window: WRef) -> Either<SlidingWindow, DiscreteWindow>;
    fn expression(&self, id: ExprId) -> &Expression;
    fn func_declaration(&self, func_name: &str) -> &FuncDecl;
}

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

#[extends_mode(IrExprTrait, IrExprMode, ir_expr)]
#[derive(Debug, Clone, HirMode)]
pub(crate) struct DepAnaMode {
    ir_expr: IrExprMode,
    dependencies: Dependencies,
}

#[mode_functionality]
pub(crate) trait DepAnaTrait {
    fn direct_accesses(&self, who: SRef) -> Vec<SRef>;

    fn transitive_accesses(&self, who: SRef) -> Vec<SRef>;

    fn direct_accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn transitive_accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)>; // (non-transitive)

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)>; // (non-transitive)

    fn graph(&self) -> &DependencyGraph;
}

#[extends_mode(IrExprTrait, IrExprMode, ir_expr)]
#[extends_mode(DepAnaTrait, DepAnaMode, dependencies)]
#[derive(Debug, Clone, HirMode)]
pub(crate) struct TypedMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    tts: TypeTable,
}

#[mode_functionality]
pub(crate) trait TypedTrait {
    fn stream_type(&self, _sr: SRef) -> HirType;
    fn is_periodic(&self, _sr: SRef) -> bool;
    fn is_event(&self, _sr: SRef) -> bool;
    fn expr_type(&self, _eid: ExprId) -> HirType;
}

#[derive(Debug, Clone)]
pub(crate) struct EvaluationOrder {
    event_layers: LayerRepresentation,
    periodic_layers: LayerRepresentation,
}
#[extends_mode(IrExprTrait, IrExprMode, ir_expr)]
#[extends_mode(DepAnaTrait, DepAnaMode, dependencies)]
#[extends_mode(TypedTrait, TypedMode, types)]
#[derive(Debug, Clone, HirMode)]
pub(crate) struct OrderedMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    types: TypedMode,
    layers: EvaluationOrder,
}
#[mode_functionality]
pub(crate) trait OrderedTrait {
    fn stream_layers(&self, sr: SRef) -> StreamLayers;
}

#[derive(Debug, Clone)]
pub(crate) struct Memory {
    memory_bound_per_stream: HashMap<SRef, MemorizationBound>,
}

#[extends_mode(IrExprTrait, IrExprMode, ir_expr)]
#[extends_mode(DepAnaTrait, DepAnaMode, dependencies)]
#[extends_mode(TypedTrait, TypedMode, types)]
#[extends_mode(OrderedTrait, OrderedMode, layers)]
#[derive(Debug, Clone, HirMode)]
pub(crate) struct MemBoundMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    types: TypedMode,
    layers: OrderedMode,
    memory: Memory,
}
#[mode_functionality]
pub(crate) trait MemBoundTrait {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound;
}

#[extends_mode(IrExprTrait, IrExprMode, ir_expr)]
#[extends_mode(DepAnaTrait, DepAnaMode, dependencies)]
#[extends_mode(TypedTrait, TypedMode, types)]
#[extends_mode(OrderedTrait, OrderedMode, layers)]
#[extends_mode(MemBoundTrait, MemBoundMode, memory)]
#[derive(Debug, Clone, HirMode)]
pub(crate) struct CompleteMode {
    ir_expr: IrExprMode,
    dependencies: DepAnaMode,
    types: TypedMode,
    layers: OrderedMode,
    memory: MemBoundMode,
}
