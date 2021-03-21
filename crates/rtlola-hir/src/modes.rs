pub trait HirMode {}

pub(crate) mod dependencies;
pub(crate) mod dg_functionality;
pub mod ir_expr;
pub(crate) mod memory_bounds;
pub(crate) mod ordering;
pub(crate) mod types;

use crate::function_lookup::FuncDecl;
use crate::hir::{DiscreteWindow, SlidingWindow};
use crate::type_check::{rtltc::StreamType, value_types::ConcreteValueType};
use itertools::Itertools;
use std::collections::HashMap;

use crate::modes::memory_bounds::MemorizationBound;
use crate::modes::ordering::StreamLayers;
use crate::{hir::expression::Expression, hir::ExprId, hir::Hir, hir::SRef, hir::WRef};

use self::{
    dependencies::{DependencyGraph, Streamdependencies, Windowdependencies},
    memory_bounds::LayerRepresentation,
    types::HirType,
};
use itertools::Either;

pub type ExpressionLookUps = HashMap<ExprId, Expression>;
pub type WindowLookUps = HashMap<WRef, Either<SlidingWindow, DiscreteWindow>>;
pub type FunctionLookUps = HashMap<String, FuncDecl>;

#[derive(Clone, Debug)]
pub struct IrExpr {
    exprid_to_expr: ExpressionLookUps,
    windows: WindowLookUps,
    func_table: FunctionLookUps,
}

#[covers_functionality(IrExprTrait, ir_expr)]
#[derive(Clone, Debug, HirMode)]
pub struct IrExprMode {
    ir_expr: IrExpr,
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
pub(crate) struct DepAna {
    direct_accesses: Streamdependencies,
    transitive_accesses: Streamdependencies,
    direct_accessed_by: Streamdependencies,
    transitive_accessed_by: Streamdependencies,
    aggregated_by: Windowdependencies,
    aggregates: Windowdependencies,
    graph: DependencyGraph,
}

#[covers_functionality(IrExprTrait, ir_expr)]
#[covers_functionality(DepAnaTrait, dependencies)]
#[derive(Debug, Clone, HirMode)]
pub struct DepAnaMode {
    ir_expr: IrExpr,
    dependencies: DepAna,
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

#[covers_functionality(IrExprTrait, ir_expr)]
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[derive(Debug, Clone, HirMode)]
pub struct TypedMode {
    ir_expr: IrExpr,
    dependencies: DepAna,
    types: Typed,
}

#[derive(Debug, Clone)]
pub struct Typed {
    stream_types: HashMap<SRef, StreamType>,
    expression_types: HashMap<ExprId, StreamType>,
    param_types: HashMap<(SRef, usize), ConcreteValueType>,
}

impl Typed {
    pub(crate) fn new(
        stream_types: HashMap<SRef, StreamType>,
        expression_types: HashMap<ExprId, StreamType>,
        param_types: HashMap<(SRef, usize), ConcreteValueType>,
    ) -> Self {
        Typed { stream_types, expression_types, param_types }
    }
}

#[mode_functionality]
pub trait TypedTrait {
    fn stream_type(&self, _sr: SRef) -> HirType;
    fn is_periodic(&self, _sr: SRef) -> bool;
    fn is_event(&self, _sr: SRef) -> bool;
    fn expr_type(&self, _eid: ExprId) -> HirType;
}

#[derive(Debug, Clone)]
pub struct Ordered {
    event_layers: LayerRepresentation,
    periodic_layers: LayerRepresentation,
}
#[covers_functionality(IrExprTrait, ir_expr)]
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(OrderedTrait, layers)]
#[derive(Debug, Clone, HirMode)]
pub struct OrderedMode {
    ir_expr: IrExpr,
    dependencies: DepAna,
    types: Typed,
    layers: Ordered,
}
#[mode_functionality]
pub(crate) trait OrderedTrait {
    fn stream_layers(&self, sr: SRef) -> StreamLayers;
}

#[derive(Debug, Clone)]
pub(crate) struct MemBound {
    memory_bound_per_stream: HashMap<SRef, MemorizationBound>,
}

#[covers_functionality(IrExprTrait, ir_expr)]
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(OrderedTrait, layers)]
#[covers_functionality(MemBoundTrait, memory)]
#[derive(Debug, Clone, HirMode)]
pub struct MemBoundMode {
    ir_expr: IrExpr,
    dependencies: DepAna,
    types: Typed,
    layers: Ordered,
    memory: MemBound,
}
#[mode_functionality]
pub(crate) trait MemBoundTrait {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound;
}

#[covers_functionality(IrExprTrait, ir_expr)]
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(OrderedTrait, layers)]
#[covers_functionality(MemBoundTrait, memory)]
#[derive(Debug, Clone, HirMode)]
pub struct CompleteMode {
    ir_expr: IrExpr,
    dependencies: DepAna,
    types: Typed,
    layers: Ordered,
    memory: MemBound,
}
