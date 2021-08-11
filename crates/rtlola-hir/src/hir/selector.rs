use crate::hir::{ConcretePacingType, Hir, Output, SRef, TypedTrait};
use crate::modes::types::HirType;
use crate::modes::HirMode;

pub trait Selectable: Copy {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub enum FilterSelector {
    Any,
    Filtered,
    Unfiltered,
}

impl Selectable for FilterSelector {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let output = hir.output(sref).expect("Asserted above");
        match self {
            FilterSelector::Any => true,
            FilterSelector::Filtered => output.instance_template.filter.is_some(),
            FilterSelector::Unfiltered => output.instance_template.filter.is_none(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CloseSelector {
    Any,
    EventBased,
    DynamicPeriodic,
    StaticPeriodic,
    AnyPeriodic,
    NotClosed,
}

impl Selectable for CloseSelector {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let output = hir.output(sref).expect("Asserted above");
        let close_ty: Option<HirType> = output.instance_template.close.map(|eid| hir.expr_type(eid));
        match self {
            CloseSelector::Any => true,
            CloseSelector::EventBased => {
                close_ty
                    .map(|t| matches!(t.pacing_ty, ConcretePacingType::Event(_) | ConcretePacingType::Constant))
                    .unwrap_or(false)
            },
            CloseSelector::DynamicPeriodic => {
                close_ty
                    .map(|t| {
                        matches!(
                            t.pacing_ty,
                            ConcretePacingType::FixedPeriodic(_) | ConcretePacingType::Periodic
                        ) && !matches!(t.spawn.0, ConcretePacingType::Constant)
                    })
                    .unwrap_or(false)
            },
            CloseSelector::StaticPeriodic => {
                close_ty
                    .map(|t| {
                        matches!(
                            t.pacing_ty,
                            ConcretePacingType::FixedPeriodic(_) | ConcretePacingType::Periodic
                        ) && matches!(t.spawn.0, ConcretePacingType::Constant)
                    })
                    .unwrap_or(false)
            },
            CloseSelector::AnyPeriodic => {
                close_ty
                    .map(|t| {
                        matches!(
                            t.pacing_ty,
                            ConcretePacingType::FixedPeriodic(_) | ConcretePacingType::Periodic
                        )
                    })
                    .unwrap_or(false)
            },
            CloseSelector::NotClosed => output.instance_template.close.is_none(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PacingSelector {
    Any,
    EventBased,
    Periodic,
}

impl PacingSelector {
    fn select_eval<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        match self {
            PacingSelector::Any => true,
            PacingSelector::EventBased => hir.is_event(sref),
            PacingSelector::Periodic => hir.is_periodic(sref),
        }
    }

    fn select_spawn<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let ty = hir.stream_type(sref).spawn.0;
        match self {
            PacingSelector::Any => true,
            PacingSelector::EventBased => matches!(ty, ConcretePacingType::Event(_)),
            PacingSelector::Periodic => {
                matches!(ty, ConcretePacingType::FixedPeriodic(_) | ConcretePacingType::Periodic)
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ParameterSelector {
    Any,
    Parameterized,
    NotParameterized,
}

impl Selectable for ParameterSelector {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let paras = &hir.output(sref).unwrap().params;
        match self {
            ParameterSelector::Any => true,
            ParameterSelector::Parameterized => !paras.is_empty(),
            ParameterSelector::NotParameterized => paras.is_empty(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct All {}
impl Selectable for All {
    fn select<M: HirMode + TypedTrait>(&self, _hir: &Hir<M>, _sref: SRef) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Static {}
impl Selectable for Static {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let spawn_ty = hir.stream_type(sref).spawn.0;
        matches!(spawn_ty, ConcretePacingType::Constant)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Dynamic {
    spawn: PacingSelector,
    close: CloseSelector,
    parameter: ParameterSelector,
}
impl Selectable for Dynamic {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let spawn_ty = hir.stream_type(sref).spawn.0;
        !matches!(spawn_ty, ConcretePacingType::Constant)
            && self.spawn.select_spawn(hir, sref)
            && self.close.select(hir, sref)
            && self.parameter.select(hir, sref)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamSelector<'a, M: HirMode + TypedTrait, S: Selectable> {
    hir: &'a Hir<M>,
    state: S,
    filter: FilterSelector,
    eval: PacingSelector,
}

impl<'a, M: HirMode + TypedTrait, S: Selectable> StreamSelector<'a, M, S> {
    pub fn filtered(&mut self) -> &mut Self {
        self.filter = FilterSelector::Filtered;
        self
    }

    pub fn unfiltered(&mut self) -> &mut Self {
        self.filter = FilterSelector::Unfiltered;
        self
    }

    pub fn filter(&mut self, selector: FilterSelector) -> &mut Self {
        self.filter = selector;
        self
    }

    pub fn periodic_eval(&mut self) -> &mut Self {
        self.eval = PacingSelector::Periodic;
        self
    }

    pub fn event_based_eval(&mut self) -> &mut Self {
        self.eval = PacingSelector::EventBased;
        self
    }

    pub fn eval(&mut self, selector: PacingSelector) -> &mut Self {
        self.eval = selector;
        self
    }

    fn select(&self, sref: SRef) -> bool {
        assert!(sref.is_output());
        self.filter.select(self.hir, sref) && self.eval.select_eval(self.hir, sref) && self.state.select(self.hir, sref)
    }

    pub fn build(self) -> Vec<&'a Output> {
        self.hir.outputs().filter(|o| self.select(o.sr)).collect()
    }
}

impl<'a, M: HirMode + TypedTrait> StreamSelector<'a, M, All> {
    pub fn all(hir: &'a Hir<M>) -> Self {
        StreamSelector {
            hir,
            state: All {},
            filter: FilterSelector::Any,
            eval: PacingSelector::Any,
        }
    }

    pub fn static_streams(self) -> StreamSelector<'a, M, Static> {
        StreamSelector {
            hir: self.hir,
            state: Static {},
            filter: self.filter,
            eval: self.eval,
        }
    }

    pub fn dynamic_streams(self) -> StreamSelector<'a, M, Dynamic> {
        StreamSelector {
            hir: self.hir,
            state: Dynamic {
                spawn: PacingSelector::Any,
                close: CloseSelector::Any,
                parameter: ParameterSelector::Any,
            },
            filter: self.filter,
            eval: self.eval,
        }
    }
}

impl<'a, M: HirMode + TypedTrait> StreamSelector<'a, M, Dynamic> {
    pub fn periodic_spawn(&mut self) -> &mut Self {
        self.state.spawn = PacingSelector::Periodic;
        self
    }

    pub fn event_based_spawn(&mut self) -> &mut Self {
        self.state.spawn = PacingSelector::EventBased;
        self
    }

    pub fn spawn(&mut self, selector: PacingSelector) -> &mut Self {
        self.state.spawn = selector;
        self
    }

    pub fn close(&mut self, selector: CloseSelector) -> &mut Self {
        self.state.close = selector;
        self
    }

    pub fn parameterized(&mut self) -> &mut Self {
        self.state.parameter = ParameterSelector::Parameterized;
        self
    }

    pub fn not_parameterized(&mut self) -> &mut Self {
        self.state.parameter = ParameterSelector::NotParameterized;
        self
    }

    pub fn parameters(&mut self, selector: ParameterSelector) -> &mut Self {
        self.state.parameter = selector;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rtlola_parser::ParserConfig;
    use rtlola_reporting::Handler;

    use super::*;
    use crate::modes::TypedMode;

    fn parse(spec: &str) -> Hir<TypedMode> {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let ast = ParserConfig::for_string(spec.into())
            .parse()
            .unwrap_or_else(|e| panic!("{}", e));
        let hir = crate::from_ast(ast, &handler).expect("Invalid AST:").analyze_dependencies(&handler).expect("Dependency error").check_types(&handler).expect("Type error");
        hir
    }

    #[test]
    fn test_dynamic(){
        let hir = parse("output x @1Hz := 42");
        assert_eq!(hir.select().static_streams().build().len(), 1);
    }
}
