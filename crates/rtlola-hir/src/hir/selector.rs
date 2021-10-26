//! This module covers a variety of selectors to extract different kinds of output streams from the High-Level Intermediate Representation (HIR) of an RTLola specification.
//!
//! The entrypoint is the [select](crate::RtLolaHir::select) method of the HIR which requires at least the Typed State.
//!
//! # Most Notable Structs and Enums
//! * [StreamSelector] is the data structure representing the combination of one or multiple selectors.
//! * [FilterSelector] represents the variants of filters a stream has.
//! * [PacingSelector] allows to select streams based on their pacing type.
//! * [CloseSelector] represents the variants of close conditions a stream has.
//! * [ParameterSelector] allows to select different parameter properties of a stream.
//!
//! # See Also
//! * [RtLolaHir](crate::RtLolaHir) the High-Level Intermediate Representation (HIR) of an RTLola specification.

use crate::hir::{ConcretePacingType, Hir, Output, SRef, TypedTrait};
use crate::modes::types::HirType;
use crate::modes::HirMode;

impl<M: HirMode + TypedTrait> Hir<M> {
    /// Creates a [StreamSelector] to query the HIR for different classes of output streams.
    pub fn select(&self) -> StreamSelector<M, All> {
        StreamSelector::all(self)
    }
}

/// Represents a selectable property of an output stream.
pub trait Selectable: Copy {
    /// Returns true if the stream represented by `sref` is accepted (selected) by the selector.
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool;
}

/// An enum used to select different filter behaviours of a stream.
#[derive(Debug, Clone, Copy)]
pub enum FilterSelector {
    /// Any stream matches this selector
    Any,
    /// Only streams *with* a filter condition match this selector
    Filtered,
    /// Only streams *without* a filter condition match this selector
    Unfiltered,
}

impl Selectable for FilterSelector {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let output = hir.output(sref).unwrap();
        match self {
            FilterSelector::Any => true,
            FilterSelector::Filtered => output.instance_template.filter.is_some(),
            FilterSelector::Unfiltered => output.instance_template.filter.is_none(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// An enum used to select different closing behaviours of a stream.
pub enum CloseSelector {
    /// Any stream matches this behaviour.
    Any,
    /// Only streams *with* a close condition match this selector.
    Closed,
    /// Only streams with an event-based close condition match this selector.
    EventBased,
    /// Only streams with an periodic close condition that is dynamically scheduled match this selector.
    /// I.e. if the close expression access a periodic stream that is spawned / dynamically created.
    DynamicPeriodic,
    /// Only streams with an periodic close condition that is statically scheduled match this selector.
    StaticPeriodic,
    /// Only streams with an periodic close condition match this selector.
    AnyPeriodic,
    /// Only streams *without* a close condition match this selector.
    NotClosed,
}

impl Selectable for CloseSelector {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let output = hir.output(sref).unwrap();
        let close_ty: Option<HirType> = output.instance_template.close.map(|eid| hir.expr_type(eid));
        match self {
            CloseSelector::Any => true,
            CloseSelector::Closed => close_ty.is_some(),
            CloseSelector::EventBased => {
                close_ty
                    .map(|t| t.pacing_ty.is_event_based() || t.pacing_ty.is_constant())
                    .unwrap_or(false)
            },
            CloseSelector::DynamicPeriodic => {
                close_ty
                    .map(|t| t.pacing_ty.is_periodic() && !t.spawn.0.is_constant())
                    .unwrap_or(false)
            },
            CloseSelector::StaticPeriodic => {
                close_ty
                    .map(|t| t.pacing_ty.is_periodic() && t.spawn.0.is_constant())
                    .unwrap_or(false)
            },
            CloseSelector::AnyPeriodic => close_ty.map(|t| t.pacing_ty.is_periodic()).unwrap_or(false),
            CloseSelector::NotClosed => output.instance_template.close.is_none(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// An enum used to select different pacing behaviours.
pub enum PacingSelector {
    /// The subject is either event-based or periodic
    Any,
    /// Only event-based subjects match this selector.
    EventBased,
    /// Only periodic subjects match this selector.
    Periodic,
}

impl PacingSelector {
    fn select_eval<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let ty: ConcretePacingType = hir.stream_type(sref).pacing_ty;
        match self {
            PacingSelector::Any => true,
            PacingSelector::EventBased => ty.is_event_based(),
            PacingSelector::Periodic => ty.is_periodic(),
        }
    }

    fn select_spawn<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        let ty: ConcretePacingType = hir.stream_type(sref).spawn.0;
        match self {
            PacingSelector::Any => true,
            PacingSelector::EventBased => ty.is_event_based(),
            PacingSelector::Periodic => ty.is_periodic(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// An enum used to select different parameter configurations of a stream.
pub enum ParameterSelector {
    /// Any stream matches this selector.
    Any,
    /// Only streams *with* parameters match this selector.
    Parameterized,
    /// Only streams *without* parameters match this selector.
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
/// A selector struct to capture all streams of the hir.
pub struct All {}
impl Selectable for All {
    fn select<M: HirMode + TypedTrait>(&self, _hir: &Hir<M>, _sref: SRef) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
/// A selector struct to capture all statically created streams of the hir.
pub struct Static {}
impl Selectable for Static {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        hir.stream_type(sref).spawn.0.is_constant()
    }
}

#[derive(Debug, Clone, Copy)]
/// A selector struct to capture all dynamically created streams of the hir.
pub struct Dynamic {
    /// Determines the pacing behaviour of the selected condition.
    spawn: PacingSelector,
    /// Determines the closing behaviour of the selected streams.
    close: CloseSelector,
    /// Determines the parameter configuration of the selected streams.
    parameter: ParameterSelector,
}
impl Selectable for Dynamic {
    fn select<M: HirMode + TypedTrait>(&self, hir: &Hir<M>, sref: SRef) -> bool {
        assert!(sref.is_output());
        !hir.stream_type(sref).spawn.0.is_constant()
            && self.spawn.select_spawn(hir, sref)
            && self.close.select(hir, sref)
            && self.parameter.select(hir, sref)
    }
}

#[derive(Debug, Clone, Copy)]
/// A StreamSelector combines different selectors to extract specific stream classes from the Hir.
/// It captures a subset of output streams, where each stream matches all selectors of the StreamSelector.
pub struct StreamSelector<'a, M: HirMode + TypedTrait, S: Selectable> {
    /// A reference to the Hir
    hir: &'a Hir<M>,
    /// The underlying selector state representing a certain class of streams.
    ///
    /// # See also:
    /// * [All]
    /// * [Static]
    /// * [Dynamic]
    state: S,
    /// Determines the filter behaviour of the selected streams.
    filter: FilterSelector,
    /// Determines the evaluation pacing of the selected streams.
    eval: PacingSelector,
}

impl<'a, M: HirMode + TypedTrait, S: Selectable> StreamSelector<'a, M, S> {
    /// Selects streams *with* a filter condition.
    pub fn filtered(mut self) -> Self {
        self.filter = FilterSelector::Filtered;
        self
    }

    /// Selects streams *without* a filter condition.
    pub fn unfiltered(mut self) -> Self {
        self.filter = FilterSelector::Unfiltered;
        self
    }

    /// Selects streams with the filter behaviour specified by the FilterSelector.
    pub fn filter(mut self, selector: FilterSelector) -> Self {
        self.filter = selector;
        self
    }

    /// Selects streams with periodic evaluation pacing.
    pub fn periodic_eval(mut self) -> Self {
        self.eval = PacingSelector::Periodic;
        self
    }

    /// Selects streams with event-based evaluation pacing.
    pub fn event_based_eval(mut self) -> Self {
        self.eval = PacingSelector::EventBased;
        self
    }

    /// Selects streams with an evaluation pacing matching the [PacingSelector].
    pub fn eval(mut self, selector: PacingSelector) -> Self {
        self.eval = selector;
        self
    }

    fn select(&self, sref: SRef) -> bool {
        assert!(sref.is_output());
        self.filter.select(self.hir, sref) && self.eval.select_eval(self.hir, sref) && self.state.select(self.hir, sref)
    }

    /// Construct the represented subset of output streams matching the given selections.
    pub fn build(self) -> impl Iterator<Item = &'a Output> {
        self.hir.outputs().filter(move |o| self.select(o.sr))
    }
}

impl<'a, M: HirMode + TypedTrait> StreamSelector<'a, M, All> {
    /// Creates a new StreamSelector matching all output streams.
    pub fn all(hir: &'a Hir<M>) -> Self {
        StreamSelector {
            hir,
            state: All {},
            filter: FilterSelector::Any,
            eval: PacingSelector::Any,
        }
    }

    /// Selects statically created streams.
    /// I.e. only streams without a spawn condition.
    pub fn static_streams(self) -> StreamSelector<'a, M, Static> {
        StreamSelector {
            hir: self.hir,
            state: Static {},
            filter: self.filter,
            eval: self.eval,
        }
    }

    /// Selects dynamically created streams.
    /// I.e. only streams with a spawn condition.
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
    /// Selects streams with a periodic spawn pacing.
    pub fn periodic_spawn(mut self) -> Self {
        self.state.spawn = PacingSelector::Periodic;
        self
    }

    /// Selects streams with a event-based spawn pacing.
    pub fn event_based_spawn(mut self) -> Self {
        self.state.spawn = PacingSelector::EventBased;
        self
    }

    /// Selects streams with a spawn pacing matching the [PacingSelector].
    pub fn spawn(mut self, selector: PacingSelector) -> Self {
        self.state.spawn = selector;
        self
    }

    /// Selects streams with a closing behaviour matching the [CloseSelector]
    pub fn close(mut self, selector: CloseSelector) -> Self {
        self.state.close = selector;
        self
    }

    /// Selects streams with parameters
    pub fn parameterized(mut self) -> Self {
        self.state.parameter = ParameterSelector::Parameterized;
        self
    }

    /// Selects streams without parameters
    pub fn not_parameterized(mut self) -> Self {
        self.state.parameter = ParameterSelector::NotParameterized;
        self
    }

    /// Selects streams with a parameter configuration matching the [ParameterSelector]
    pub fn parameters(mut self, selector: ParameterSelector) -> Self {
        self.state.parameter = selector;
        self
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rtlola_parser::ParserConfig;

    use super::*;
    use crate::CompleteMode;

    macro_rules! assert_streams {
        ($streams:expr, $expected:expr) => {
            let names: Vec<&str> = $streams.map(|o| o.name.as_str()).sorted().collect();
            let expect: Vec<&str> = $expected.into_iter().sorted().collect();
            assert_eq!(names, expect);
        };
    }

    fn get_hir() -> Hir<CompleteMode> {
        let spec = "input i: Int8\n\
            output a @1Hz := true\n\
            output b @1Hz spawn if i = 5 filter i.hold(or: 0) = 5 := 42\n\
            output c (p) spawn @1Hz with i.hold(or: 0) := i + p\n\
            output d spawn if i = 8 close a := i\n\
            output e @1Hz spawn if i = 3 filter i.hold(or: 1) % 2 = 0 close e := true\n\
            output f (p: Int8) spawn @1Hz with i.aggregate(over:1s, using: sum) close i = 8 := i\n\
            output g filter i = 5 := i + 5";

        let ast = ParserConfig::for_string(spec.into())
            .parse()
            .unwrap_or_else(|e| panic!("{:?}", e));
        let hir = crate::fully_analyzed(ast).expect("Invalid Spec");
        hir
    }

    #[test]
    fn test_all() {
        let hir = get_hir();

        assert_streams!(hir.select().build(), vec!["a", "b", "c", "d", "e", "f", "g"]);

        assert_streams!(hir.select().periodic_eval().build(), vec!["a", "b", "e"]);
        assert_streams!(hir.select().event_based_eval().build(), vec!["c", "d", "f", "g"]);
        assert_streams!(hir.select().eval(PacingSelector::Periodic).build(), vec!["a", "b", "e"]);
        assert_streams!(
            hir.select().eval(PacingSelector::EventBased).build(),
            vec!["c", "d", "f", "g"]
        );

        assert_streams!(hir.select().filtered().build(), vec!["b", "e", "g"]);
        assert_streams!(hir.select().unfiltered().build(), vec!["a", "c", "d", "f"]);
        assert_streams!(
            hir.select().filter(FilterSelector::Filtered).build(),
            vec!["b", "e", "g"]
        );
        assert_streams!(
            hir.select().filter(FilterSelector::Unfiltered).build(),
            vec!["a", "c", "d", "f"]
        );
    }

    #[test]
    fn test_dynamic() {
        let hir = get_hir();

        assert_streams!(hir.select().dynamic_streams().build(), vec!["b", "c", "d", "e", "f"]);
        assert_streams!(hir.select().dynamic_streams().periodic_eval().build(), vec!["b", "e"]);
        assert_streams!(
            hir.select().dynamic_streams().event_based_eval().build(),
            vec!["c", "d", "f"]
        );
        assert_streams!(
            hir.select().dynamic_streams().eval(PacingSelector::Periodic).build(),
            vec!["b", "e"]
        );
        assert_streams!(
            hir.select().dynamic_streams().eval(PacingSelector::EventBased).build(),
            vec!["c", "d", "f"]
        );

        assert_streams!(hir.select().dynamic_streams().filtered().build(), vec!["b", "e"]);
        assert_streams!(hir.select().dynamic_streams().unfiltered().build(), vec!["c", "d", "f"]);
        assert_streams!(
            hir.select().dynamic_streams().filter(FilterSelector::Filtered).build(),
            vec!["b", "e"]
        );
        assert_streams!(
            hir.select()
                .dynamic_streams()
                .filter(FilterSelector::Unfiltered)
                .build(),
            vec!["c", "d", "f"]
        );

        assert_streams!(hir.select().dynamic_streams().periodic_spawn().build(), vec!["c", "f"]);
        assert_streams!(
            hir.select().dynamic_streams().event_based_spawn().build(),
            vec!["b", "d", "e"]
        );
        assert_streams!(
            hir.select().dynamic_streams().spawn(PacingSelector::Periodic).build(),
            vec!["c", "f"]
        );
        assert_streams!(
            hir.select().dynamic_streams().spawn(PacingSelector::EventBased).build(),
            vec!["b", "d", "e"]
        );

        assert_streams!(hir.select().dynamic_streams().parameterized().build(), vec!["c", "f"]);
        assert_streams!(
            hir.select().dynamic_streams().not_parameterized().build(),
            vec!["b", "d", "e"]
        );
        assert_streams!(
            hir.select()
                .dynamic_streams()
                .parameters(ParameterSelector::Parameterized)
                .build(),
            vec!["c", "f"]
        );
        assert_streams!(
            hir.select()
                .dynamic_streams()
                .parameters(ParameterSelector::NotParameterized)
                .build(),
            vec!["b", "d", "e"]
        );
    }

    #[test]
    fn test_close() {
        let hir = get_hir();

        assert_streams!(
            hir.select().dynamic_streams().close(CloseSelector::Closed).build(),
            vec!["d", "e", "f"]
        );

        assert_streams!(
            hir.select().dynamic_streams().close(CloseSelector::AnyPeriodic).build(),
            vec!["d", "e"]
        );

        assert_streams!(
            hir.select()
                .dynamic_streams()
                .close(CloseSelector::DynamicPeriodic)
                .build(),
            vec!["e"]
        );

        assert_streams!(
            hir.select()
                .dynamic_streams()
                .close(CloseSelector::StaticPeriodic)
                .build(),
            vec!["d"]
        );

        assert_streams!(
            hir.select().dynamic_streams().close(CloseSelector::EventBased).build(),
            vec!["f"]
        );

        assert_streams!(
            hir.select().dynamic_streams().close(CloseSelector::NotClosed).build(),
            vec!["b", "c"]
        );
    }

    #[test]
    fn test_static() {
        let hir = get_hir();

        assert_streams!(hir.select().static_streams().build(), vec!["a", "g"]);
        assert_streams!(hir.select().static_streams().periodic_eval().build(), vec!["a"]);
        assert_streams!(hir.select().static_streams().event_based_eval().build(), vec!["g"]);
        assert_streams!(
            hir.select().static_streams().eval(PacingSelector::Periodic).build(),
            vec!["a"]
        );
        assert_streams!(
            hir.select().static_streams().eval(PacingSelector::EventBased).build(),
            vec!["g"]
        );

        assert_streams!(hir.select().static_streams().filtered().build(), vec!["g"]);
        assert_streams!(hir.select().static_streams().unfiltered().build(), vec!["a"]);
        assert_streams!(
            hir.select().static_streams().filter(FilterSelector::Filtered).build(),
            vec!["g"]
        );
        assert_streams!(
            hir.select().static_streams().filter(FilterSelector::Unfiltered).build(),
            vec!["a"]
        );
    }
}
