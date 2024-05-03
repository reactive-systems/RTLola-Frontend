use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::iter::FromIterator;

use rtlola_parser::ast::{InstanceOperation, WindowOperation};
use rtlola_reporting::{RtLolaError, Span};

use crate::features::{
    Closed, DiscreteWindows, Filtered, InstanceAggregations, MultipleEvals, Parameterized, Periodics, SlidingWindows,
    Spawned, ValueTypes,
};
use crate::hir::{
    AnnotatedPacingType, ConcretePacingType, ConcreteValueType, DiscreteAggr, Expression, ExpressionKind, FnExprKind,
    Input, InstanceAggregation, Output, SlidingAggr, StreamAccessKind, StreamType, TypedTrait, WRef, WidenExprKind,
    Window,
};
use crate::{CompleteMode, RtLolaHir};

/// This trait characterizes an RTLola feature by classifying when to exclude different language constructs.
pub trait Feature {
    /// The name of this feature.
    fn name(&self) -> &'static str;

    /// Specifies whether to exclude a given Input.
    /// If the input contains constructs part of the feature an Err should be returned describing which construct was used.
    fn exclude_input(&self, _input: &Input) -> Result<(), RtLolaError> {
        Ok(())
    }

    /// Specifies whether to exclude a given Output
    /// If the output contains constructs part of the feature an Err should be returned describing which construct was used.
    fn exclude_output(&self, _output: &Output) -> Result<(), RtLolaError> {
        Ok(())
    }

    /// Specifies whether to exclude a given discrete window.
    /// The given span corresponds to the code range where the window was defined.
    /// If the discrete window contains constructs part of the feature an Err should be returned describing which construct was used.
    fn exclude_discrete_window(&self, _span: &Span, _window: &Window<DiscreteAggr>) -> Result<(), RtLolaError> {
        Ok(())
    }

    /// Specifies whether to exclude a given sliding window.
    /// The given span corresponds to the code range where the window was defined.
    /// If the sliding window contains constructs part of the feature an Err should be returned describing which construct was used.
    fn exclude_sliding_window(&self, _span: &Span, _window: &Window<SlidingAggr>) -> Result<(), RtLolaError> {
        Ok(())
    }

    /// Specifies whether to exclude a given instance aggregation.
    /// The given span corresponds to the code range where the instance aggregation was defined.
    /// If the instance aggregation contains constructs part of the feature an Err should be returned describing which construct was used.
    fn exclude_instance_aggregation(
        &self,
        _span: &Span,
        _aggregation: &InstanceAggregation,
    ) -> Result<(), RtLolaError> {
        Ok(())
    }

    /// Specifies whether to exclude a given value type
    /// If the value type is part of the feature an Err should be returned describing which value type was used.
    /// Recursing into subtypes is handled externally and does not need to be handled.
    fn exclude_value_type(&self, _span: &Span, _ty: &ConcreteValueType) -> Result<(), RtLolaError> {
        Ok(())
    }

    /// Specifies whether to exclude a given pacing type
    /// If the pacing type is part of the feature an Err should be returned describing which pacing type was used.
    fn exclude_pacing_type(&self, _span: &Span, _ty: &ConcretePacingType) -> Result<(), RtLolaError> {
        Ok(())
    }

    /// Specifies whether to exclude a certain kind of expression
    /// If the given expression kind is part of the feature an Err should be returned describing which expression kind was used.
    /// Recursing into subexpressions is handled externally and does not need to be handled.
    fn exclude_expression_kind(&self, _span: &Span, _kind: &ExpressionKind) -> Result<(), RtLolaError> {
        Ok(())
    }
}

#[allow(missing_debug_implementations)]
/// The [FeatureSelector] allows a backend to specify which RTLola Features it supports.
/// It validates, that a given Hir does not contain any of those features and returns an Error otherwise.
/// When all features are specified call [FeatureSelector::build] to perform the validation.
pub struct FeatureSelector {
    hir: RtLolaHir<CompleteMode>,
    features: Vec<Box<dyn Feature>>,
}

impl Debug for FeatureSelector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let feature_strs: Vec<String> = self.features.iter().map(|f| f.name().to_string()).collect();
        write!(f, "FeatureSelector{{hir: {:?}, features: {feature_strs:?}}}", self.hir)
    }
}

impl Feature for FeatureSelector {
    fn name(&self) -> &'static str {
        "FeatureSelector"
    }

    fn exclude_input(&self, input: &Input) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_input(input))
    }

    fn exclude_output(&self, output: &Output) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_output(output))
    }

    fn exclude_discrete_window(&self, span: &Span, window: &Window<DiscreteAggr>) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_discrete_window(span, window))
    }

    fn exclude_sliding_window(&self, span: &Span, window: &Window<SlidingAggr>) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_sliding_window(span, window))
    }

    fn exclude_instance_aggregation(&self, span: &Span, aggregation: &InstanceAggregation) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_instance_aggregation(span, aggregation))
    }

    fn exclude_value_type(&self, span: &Span, ty: &ConcreteValueType) -> Result<(), RtLolaError> {
        let current = self.iter_features(|f| f.exclude_value_type(span, ty));
        let other = match ty {
            ConcreteValueType::Bool
            | ConcreteValueType::Integer8
            | ConcreteValueType::Integer16
            | ConcreteValueType::Integer32
            | ConcreteValueType::Integer64
            | ConcreteValueType::UInteger8
            | ConcreteValueType::UInteger16
            | ConcreteValueType::UInteger32
            | ConcreteValueType::UInteger64
            | ConcreteValueType::Float32
            | ConcreteValueType::Float64
            | ConcreteValueType::TString
            | ConcreteValueType::Byte => Ok(()), /* handled by first disjunct */
            ConcreteValueType::Tuple(children) => {
                children
                    .iter()
                    .flat_map(|ty| self.exclude_value_type(span, ty).map_err(|e| e.into_iter()).err())
                    .flatten()
                    .collect::<RtLolaError>()
                    .into()
            },
            ConcreteValueType::Option(ty) => self.exclude_value_type(span, ty.as_ref()),
        };
        let mut res = RtLolaError::new();
        if let Err(e) = current {
            res.join(e);
        }
        if let Err(e) = other {
            res.join(e)
        }
        res.into()
    }

    fn exclude_pacing_type(&self, span: &Span, ty: &ConcretePacingType) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_pacing_type(span, ty))
    }

    fn exclude_expression_kind(&self, span: &Span, kind: &ExpressionKind) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_expression_kind(span, kind))
    }
}

//public interface
impl FeatureSelector {
    /// Creates a new [FeatureSelector] for the given [RtLolaHir].
    pub fn new(hir: RtLolaHir<CompleteMode>) -> Self {
        Self { hir, features: vec![] }
    }

    /// Exclude a custom feature.
    pub fn exclude_feature<F: Feature + 'static>(mut self, feature: F) -> Self {
        self.features.push(Box::new(feature));
        self
    }

    /// Constrain the specification to Lola 1.0 features.
    pub fn lola_1(self) -> Self {
        self.non_parameterized()
            .no_spawn()
            .no_filter()
            .no_close()
            .no_multiple_evals()
            .no_discrete_windows()
            .no_sliding_windows()
            .no_instance_aggregation()
            .non_periodic()
    }

    /// Constraint the specification to Lola 2.o features.
    pub fn lola_2(self) -> Self {
        self.no_discrete_windows().no_sliding_windows().non_periodic()
    }

    /// Asserts that the specification does not contain a parameterized output stream
    pub fn non_parameterized(mut self) -> Self {
        self.features.push(Box::new(Parameterized {}));
        self
    }

    /// Asserts that the specification does not contain a spawned output stream
    pub fn no_spawn(mut self) -> Self {
        self.features.push(Box::new(Spawned {}));
        self
    }

    /// Asserts that the specification does not contain a filtered output stream
    pub fn no_filter(mut self) -> Self {
        self.features.push(Box::new(Filtered {}));
        self
    }

    /// Asserts that the specification does not contain a closed output stream
    pub fn no_close(mut self) -> Self {
        self.features.push(Box::new(Closed {}));
        self
    }

    /// Restricts the specification to a non periodic fragment
    pub fn non_periodic(mut self) -> Self {
        self.features.push(Box::new(Periodics {}));
        self
    }

    /// Asserts that the specification does not contain sliding windows
    pub fn no_sliding_windows(mut self) -> Self {
        self.features.push(Box::<SlidingWindows>::default());
        self
    }

    /// Asserts that the specification does not contain discrete windows
    pub fn no_discrete_windows(mut self) -> Self {
        self.features.push(Box::<DiscreteWindows>::default());
        self
    }

    /// Asserts that the specification does not contain instance aggregations
    pub fn no_instance_aggregation(mut self) -> Self {
        self.features.push(Box::<InstanceAggregations>::default());
        self
    }

    /// Restricts the specification to not contain any of the given value types.
    pub fn not_value_type(mut self, types: &[ConcreteValueType]) -> Self {
        self.features
            .push(Box::new(ValueTypes::new(HashSet::from_iter(types.to_vec()))));
        self
    }

    /// Restricts the specification to not contain any of the window operations.
    pub fn not_window_op(mut self, ops: &[WindowOperation]) -> Self {
        let set = HashSet::from_iter(ops.to_vec());
        self.features.push(Box::new(SlidingWindows::new(set.clone())));
        self.features.push(Box::new(DiscreteWindows::new(set)));
        self
    }

    /// Restricts the specification to not contain output streams with multiple eval clauses.
    pub fn no_multiple_evals(mut self) -> Self {
        self.features.push(Box::new(MultipleEvals {}));
        self
    }

    /// Restricts the specification to not contain any of the instance operations.
    pub fn not_instance_op(mut self, ops: &[InstanceOperation]) -> Self {
        let set = HashSet::from_iter(ops.to_vec());
        self.features.push(Box::new(InstanceAggregations::new(set)));
        self
    }

    /// Validates the [RtLolaHir] against the given feature set.
    pub fn build(self) -> Result<RtLolaHir<CompleteMode>, RtLolaError> {
        let mut res = RtLolaError::new();
        self.hir.inputs().for_each(|i| {
            let ty = self.hir.stream_type(i.sr);
            if let Err(e) = self.exclude_input(i) {
                res.join(e);
            }
            if let Err(e) = self.exclude_value_type(&i.span, &ty.value_ty) {
                res.join(e);
            }
        });

        self.hir.outputs.iter().for_each(|o| {
            let ty = self.hir.stream_type(o.sr);
            let spawn_span: Span = o
                .spawn
                .as_ref()
                .map(|spawn| {
                    spawn
                        .expression
                        .map(|expr| self.hir.expression(expr).span)
                        .or_else(|| spawn.condition.map(|expr| self.hir.expression(expr).span))
                        .unwrap_or_else(|| {
                            match spawn.pacing {
                                AnnotatedPacingType::GlobalFrequency(f) => f.span,
                                AnnotatedPacingType::LocalFrequency(f) => f.span,
                                AnnotatedPacingType::UnspecifiedFrequency(f) => f.span,
                                AnnotatedPacingType::Expr(eid) => self.hir.expression(eid).span,
                                AnnotatedPacingType::NotAnnotated => Span::Unknown,
                            }
                        })
                })
                .unwrap_or(Span::Unknown);

            if let Err(e) = self.exclude_output(o) {
                res.join(e);
            }
            if let Err(e) = self.exclude_value_type(&o.span, &ty.value_ty) {
                res.join(e);
            }
            if let Err(e) = self.exclude_pacing_type(&o.span, &ty.eval_pacing) {
                res.join(e);
            }
            if let Err(e) = self.exclude_pacing_type(&spawn_span, &ty.spawn_pacing) {
                res.join(e);
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.spawn_expr(o.sr)) {
                res.join(e);
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.spawn_cond(o.sr)) {
                res.join(e);
            }
            for eval in self.hir.eval_unchecked(o.sr) {
                if let Err(e) = self.exclude_expression(eval.expression) {
                    res.join(e);
                }
                if let Err(e) = self.exclude_expression_opt(eval.condition) {
                    res.join(e);
                }
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.close_cond(o.sr)) {
                res.join(e);
            }
        });
        self.hir.discrete_windows().iter().for_each(|window| {
            let span = self.find_window_span(window.reference);
            if let Err(e) = self.exclude_discrete_window(&span, window) {
                res.join(e);
            }
        });
        self.hir.sliding_windows().iter().for_each(|window| {
            let span = self.find_window_span(window.reference);
            if let Err(e) = self.exclude_sliding_window(&span, window) {
                res.join(e);
            }
        });
        self.hir.instance_aggregations().iter().for_each(|aggr| {
            let span = self.find_window_span(aggr.reference);
            if let Err(e) = self.exclude_instance_aggregation(&span, aggr) {
                res.join(e);
            }
        });

        Result::from(res).map(|_| self.hir)
    }
}

// Private interface
impl FeatureSelector {
    fn find_window_span(&self, window: WRef) -> Span {
        fn find_access_expr(expr: &Expression, window: WRef) -> Option<Span> {
            match &expr.kind {
                ExpressionKind::StreamAccess(_, StreamAccessKind::SlidingWindow(w), _) if *w == window => {
                    Some(expr.span)
                },
                ExpressionKind::StreamAccess(_, StreamAccessKind::DiscreteWindow(w), _) if *w == window => {
                    Some(expr.span)
                },
                ExpressionKind::StreamAccess(_, StreamAccessKind::InstanceAggregation(w), _) if *w == window => {
                    Some(expr.span)
                },
                ExpressionKind::StreamAccess(_, _, _)
                | ExpressionKind::LoadConstant(_)
                | ExpressionKind::ParameterAccess(_, _) => None,
                ExpressionKind::Function(FnExprKind {
                    name: _,
                    args,
                    type_param: _,
                })
                | ExpressionKind::Tuple(args)
                | ExpressionKind::ArithLog(_, args) => args.iter().filter_map(|e| find_access_expr(e, window)).next(),
                ExpressionKind::Ite {
                    condition,
                    consequence,
                    alternative,
                } => {
                    find_access_expr(condition.as_ref(), window)
                        .or_else(|| find_access_expr(consequence.as_ref(), window))
                        .or_else(|| find_access_expr(alternative.as_ref(), window))
                },
                ExpressionKind::Widen(WidenExprKind { expr: target, ty: _ })
                | ExpressionKind::TupleAccess(target, _) => find_access_expr(target.as_ref(), window),
                ExpressionKind::Default { expr, default } => {
                    find_access_expr(expr.as_ref(), window).or_else(|| find_access_expr(default.as_ref(), window))
                },
            }
        }

        let caller_ref = match window {
            WRef::Sliding(_) => self.hir.single_sliding(window).caller,
            WRef::Discrete(_) => self.hir.single_discrete(window).caller,
            WRef::Instance(_) => self.hir.single_instance_aggregation(window).caller,
        };
        let caller = self
            .hir
            .output(caller_ref)
            .expect("Only output streams can call a window");

        let spawn = caller.spawn.as_ref().and_then(|spawn| {
            spawn
                .condition
                .and_then(|expr| find_access_expr(self.hir.expression(expr), window))
                .or_else(|| {
                    spawn
                        .expression
                        .and_then(|expr| find_access_expr(self.hir.expression(expr), window))
                })
        });
        let eval = caller
            .eval
            .iter()
            .find_map(|eval| find_access_expr(self.hir.expression(eval.expr), window))
            .or_else(|| {
                caller
                    .eval
                    .iter()
                    .flat_map(|eval| eval.condition)
                    .find_map(|expr| find_access_expr(self.hir.expression(expr), window))
            });
        let close = caller
            .close
            .as_ref()
            .and_then(|close| find_access_expr(self.hir.expression(close.condition), window));

        spawn.or(eval).or(close).unwrap_or(Span::Unknown)
    }

    fn iter_features<F>(&self, op: F) -> Result<(), RtLolaError>
    where
        F: Fn(&Box<dyn Feature>) -> Result<(), RtLolaError>,
    {
        self.features
            .iter()
            .flat_map(|f| op(f).map_err(|e| e.into_iter()).err())
            .flatten()
            .collect::<RtLolaError>()
            .into()
    }

    fn exclude_expression_opt(&self, exp: Option<&Expression>) -> Result<(), RtLolaError> {
        exp.map(|e| self.exclude_expression(e)).unwrap_or(Ok(()))
    }

    /// Recursively walk expression ast.
    fn exclude_expression(&self, exp: &Expression) -> Result<(), RtLolaError> {
        let span = &exp.span;
        let stream_ty: StreamType = self.hir.expr_type(exp.eid);
        let mut res = RtLolaError::new();
        if let Err(e) = self.exclude_value_type(span, &stream_ty.value_ty) {
            res.join(e);
        }
        if let Err(e) = self.exclude_expression_kind(span, &exp.kind) {
            res.join(e);
        }
        match &exp.kind {
            ExpressionKind::ParameterAccess(_, _) | ExpressionKind::LoadConstant(_) => {},
            ExpressionKind::Function(FnExprKind {
                name: _,
                args: sub_exps,
                type_param: _,
            })
            | ExpressionKind::Tuple(sub_exps)
            | ExpressionKind::StreamAccess(_, _, sub_exps)
            | ExpressionKind::ArithLog(_, sub_exps) => {
                sub_exps.iter().for_each(|exp| {
                    if let Err(e) = self.exclude_expression(exp) {
                        res.join(e)
                    }
                })
            },
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                if let Err(e) = self.exclude_expression(condition.as_ref()) {
                    res.join(e);
                }
                if let Err(e) = self.exclude_expression(consequence.as_ref()) {
                    res.join(e);
                }
                if let Err(e) = self.exclude_expression(alternative.as_ref()) {
                    res.join(e)
                }
            },
            ExpressionKind::TupleAccess(target, _) => {
                if let Err(e) = self.exclude_expression(target.as_ref()) {
                    res.join(e);
                }
            },
            ExpressionKind::Widen(WidenExprKind { expr, ty: _ }) => {
                if let Err(e) = self.exclude_expression(expr.as_ref()) {
                    res.join(e);
                }
            },
            ExpressionKind::Default { expr, default } => {
                if let Err(e) = self.exclude_expression(expr.as_ref()) {
                    res.join(e);
                }
                if let Err(e) = self.exclude_expression(default.as_ref()) {
                    res.join(e);
                }
            },
        };

        res.into()
    }
}

#[cfg(test)]
mod test {
    use rtlola_parser::ast::WindowOperation;
    use rtlola_parser::ParserConfig;
    use rtlola_reporting::Handler;

    use crate::fully_analyzed;
    use crate::hir::{ConcreteValueType, FeatureSelector};

    fn builder(cfg: &ParserConfig) -> (FeatureSelector, Handler) {
        use rtlola_parser::parse;
        let handler = Handler::from(cfg);
        let ast = parse(&cfg).map_err(|e| handler.emit_error(&e)).unwrap();
        let hir = fully_analyzed(ast).map_err(|e| handler.emit_error(&e)).unwrap();
        (FeatureSelector::new(hir), handler)
    }

    #[test]
    fn parameterized() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c(p: UInt)
                spawn with (a)
                eval with b
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);
        assert_eq!(
            builder
                .non_parameterized()
                .build()
                .map_err(|e| e.num_errors())
                .unwrap_err(),
            1
        );
    }

    #[test]
    fn spawned() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c
                spawn when b
                eval with b
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);
        assert_eq!(builder.no_spawn().build().map_err(|e| e.num_errors()).unwrap_err(), 1);
    }

    #[test]
    fn filtered() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c
                eval with a when b
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);
        assert_eq!(builder.no_filter().build().map_err(|e| e.num_errors()).unwrap_err(), 1);
    }

    #[test]
    fn closed() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c
                eval with a
                close when b
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);
        assert_eq!(builder.no_close().build().map_err(|e| e.num_errors()).unwrap_err(), 1);
    }

    #[test]
    fn periodics() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c @1Hz := b.hold(or: false)
            trigger c
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(
            builder.non_periodic().build().map_err(|e| e.num_errors()).unwrap_err(),
            2
        );
    }

    #[test]
    fn sliding_windows() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c @1Hz := a.aggregate(over: 5s, using: sum)\n\
            output d @1Hz := b.aggregate(over: 5s, using: count)
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(
            builder
                .no_sliding_windows()
                .build()
                .map_err(|e| e.num_errors())
                .unwrap_err(),
            2
        );
    }

    #[test]
    fn discrete_window() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c := a.aggregate(over_discrete: 5, using: sum)\n\
            output d := b.aggregate(over_discrete: 5, using: count)
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(
            builder
                .no_discrete_windows()
                .build()
                .map_err(|e| e.num_errors())
                .unwrap_err(),
            2
        );
    }

    #[test]
    fn instance_aggregation() {
        let spec = "\
            input a: Int32\n\
            output b (p) spawn with a eval when a > 5 with b(p).offset(by: -1).defaults(to: 0) + 1\n\
            output c eval with b.aggregate(over_instances: fresh, using: Σ)\n
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(
            builder
                .no_instance_aggregation()
                .build()
                .map_err(|e| e.num_errors())
                .unwrap_err(),
            1
        );
    }

    #[test]
    fn window_op() {
        let spec = "\
            input a: UInt\n\
            input b: Bool\n\
            output c @1Hz := a.aggregate(over: 5s, using: sum)\n\
            output d := b.aggregate(over_discrete: 5, using: count)
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(
            builder
                .not_window_op(&[WindowOperation::Sum, WindowOperation::Count])
                .build()
                .map_err(|e| e.num_errors())
                .unwrap_err(),
            2
        );
    }

    #[test]
    fn value_type() {
        let spec = "\
            input a: Float\n\
            input b: Bool\n\
            output c @1Hz := a.aggregate(over: 5s, using: avg).defaults(to: 0.0)\n\
            output d @a := 42.0 = 0.0
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(
            builder
                .not_value_type(&[ConcreteValueType::Float32, ConcreteValueType::Float64])
                .build()
                .map_err(|e| e.num_errors())
                .unwrap_err(),
            7
        );
    }

    #[test]
    fn lola_1() {
        let spec = "\
            input a: Float\n\
            input b: Bool\n\
            output c @1Hz := a.aggregate(over: 5s, using: avg).defaults(to: 0.0)\n\
            output d(p: Bool)\n\
                spawn with b\n\
                eval with b == p when a == 42.0\n\
                close when b
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(builder.lola_1().build().map_err(|e| e.num_errors()).unwrap_err(), 6);
    }

    #[test]
    fn lola_2() {
        let spec = "\
            input a: Float\n\
            input b: Bool\n\
            output c @1Hz := a.aggregate(over: 5s, using: avg).defaults(to: 0.0)\n\
            output d(p: Bool)\n\
                spawn with b\n\
                eval with b == p when a == 42.0\n\
                close when b
        ";
        let config = ParserConfig::for_string(spec.into());
        let (builder, _handler) = builder(&config);

        assert_eq!(builder.lola_2().build().map_err(|e| e.num_errors()).unwrap_err(), 2);
    }
}
