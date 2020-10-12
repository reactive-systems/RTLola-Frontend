use crate::common_ir::{MemorizationBound, SRef};
use crate::hir::modes::dependencies::{DependenciesAnalyzed, DependenciesWrapper};
use crate::hir::modes::ir_expr::{IrExprWrapper, WithIrExpr};
use crate::hir::modes::types::{TypeChecked, TypedWrapper};
use crate::hir::modes::Complete;
use crate::hir::modes::Dependencies;
use crate::hir::modes::MemoryAnalyzed;
use crate::hir::StreamReference;
use crate::{common_ir::Tracking, hir::Window};
use crate::{hir::Hir, mir, mir::Mir};

use super::{IrExpression, Typed};

impl Hir<Complete> {
    pub(crate) fn lower(self) -> Mir {
        let outputs = self
            .outputs
            .clone()
            .into_iter()
            .map(|o| {
                let sr = o.sr;
                mir::OutputStream {
                    name: o.name,
                    ty: Self::lower_type(sr, &self.mode),
                    expr: self.lower_expr(sr),
                    input_dependencies: self.accesses(sr).into_iter().filter(SRef::is_input).collect(), // TODO: Is this supposed to be transitive?
                    outgoing_dependencies: self.accesses(sr).into_iter().filter(|_sr| todo!()).collect(), // TODO: Is this supposed to be transitive?
                    dependent_streams: self.accessed_by(sr).into_iter().map(Self::lower_dependency).collect(),
                    dependent_windows: self.aggregated_by(sr).into_iter().map(|(_sr, wr)| wr).collect(),
                    memory_bound: self.memory(sr),
                    layer: self.layer(sr),
                    reference: sr,
                }
            })
            .collect();
        let event_driven = self
            .outputs
            .iter()
            .filter(|o| self.is_event(o.sr))
            .map(|o| mir::EventDrivenStream { reference: o.sr })
            .collect();
        let time_driven =
            self.outputs.iter().filter(|o| self.is_periodic(o.sr)).map(|o| self.lower_periodic(o.sr)).collect();
        let sliding_windows = self.windows().iter().map(|win| self.lower_window(*win)).collect();
        let Hir { inputs, triggers, mode, .. } = self;
        let inputs = inputs
            .into_iter()
            .map(|i| {
                let sr = i.sr;
                mir::InputStream {
                    name: i.name,
                    ty: Self::lower_type(sr, &mode),
                    // dependent_streams: mode.accessed_by(sr).into_iter().map(|sr| Self::lower_dependency(*sr)).collect(),
                    dependent_streams: mode.accessed_by(sr).to_vec(),
                    dependent_windows: mode.aggregated_by(sr).to_vec(),
                    layer: mode.layer(sr),
                    memory_bound: mode.memory(sr),
                    reference: sr,
                }
            })
            .collect();
        let triggers = triggers.into_iter().map(|t| mir::Trigger { message: t.message, reference: t.sr }).collect();
        Mir { inputs, outputs, triggers, event_driven, time_driven, sliding_windows }
    }

    fn lower_periodic(&self, _sr: StreamReference) -> mir::TimeDrivenStream {
        todo!()
    }

    fn lower_window(&self, _win: Window) -> mir::SlidingWindow {
        todo!()
    }

    fn lower_type(_sr: StreamReference, _mode: &Complete) -> mir::Type {
        unimplemented!()
    }

    fn lower_dependency(_dep: StreamReference) -> Tracking {
        todo!()
    }

    fn lower_expr(&self, _sr: StreamReference) -> mir::Expression {
        todo!()
    }
}

impl DependenciesWrapper for Complete {
    type InnerD = Dependencies;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl MemoryAnalyzed for Complete {
    fn memory(&self, _sr: StreamReference) -> MemorizationBound {
        todo!()
    }
}

impl TypedWrapper for Complete {
    type InnerT = Typed;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

impl IrExprWrapper for Complete {
    type InnerE = IrExpression;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}
