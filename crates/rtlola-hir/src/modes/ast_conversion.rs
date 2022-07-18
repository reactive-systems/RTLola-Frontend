mod naming;

use std::collections::HashMap;
use std::convert::TryInto;
use std::rc::Rc;
use std::time::Duration;

use rtlola_parser::ast::{
    self, FunctionName, Literal as AstLiteral, NodeId, RtLolaAst, SpawnSpec, StreamAccessKind, Type,
};
use rtlola_reporting::{Diagnostic, RtLolaError, Span};
use serde::{Deserialize, Serialize};

use super::BaseMode;
use crate::hir::{
    AnnotatedPacingType, AnnotatedType, CloseTemplate, Constant as HirConstant, DiscreteAggr, EvalTemplate, ExprId,
    Expression, ExpressionKind, ExpressionMaps, FnExprKind, Hir, Inlined, Input, Literal, Offset, Output, Parameter,
    SRef, SlidingAggr, SpawnTemplate, StreamAccessKind as IRAccess, Trigger, WRef, WidenExprKind, Window,
};
use crate::modes::ast_conversion::naming::{Declaration, NamingAnalysis};
use crate::stdlib::FuncDecl;

impl Hir<BaseMode> {
    /// Transforms a [RtLolaAst] into an [Hir]. The `handler` is provided to the [NamingAnalysis] to perform its validity analysis.
    /// Returns an Hir instance or an [TransformationErr] for error reporting.
    ///
    /// # Procedure
    /// - Performs the [NamingAnalysis]
    /// - Checks for proper expression kinds within all expressions.
    /// - Ensures no missing expressions and inlines all constant definitions, see [Constant](crate::hir:Constant).
    /// - Assigns new expression ids to all expressions.
    pub(crate) fn from_ast(ast: RtLolaAst) -> Result<Self, RtLolaError> {
        let mut naming_analyzer = NamingAnalysis::new();
        let decl_table = naming_analyzer.check(&ast)?;
        let func_table: HashMap<String, FuncDecl> = decl_table
            .values()
            .filter(|decl| matches!(decl, Declaration::Func(_)))
            .map(|decl| {
                if let Declaration::Func(fun_decl) = decl {
                    (fun_decl.name.name.clone(), (**fun_decl).clone())
                } else {
                    unreachable!("assured by filter")
                }
            })
            .collect();

        let mut stream_by_name = HashMap::new();

        for (ix, o) in ast.outputs.iter().enumerate() {
            let sr = SRef::Out(ix);
            stream_by_name.insert(o.name.name.clone(), sr);
        }
        for (ix, i) in ast.inputs.iter().enumerate() {
            let sr = SRef::In(ix);
            stream_by_name.insert(i.name.name.clone(), sr);
        }
        let stream_by_name = stream_by_name;
        ExpressionTransformer::run(decl_table, stream_by_name, ast, func_table).map_err(|e| e.into_diagnostic().into())
    }
}

/// The Hir Spawn definition is composed of two optional expressions and the annotated pacing.
/// The first one refers to the spawn target while the second one represents the spawn condition.
#[derive(Debug, Clone, Copy)]
pub struct SpawnDef<'a> {
    /// The target expression of the spawn, e.g. spawn with (3,x)
    pub target: Option<&'a Expression>,
    /// The conditional expression of the spawn, e.g. when x > 5
    pub condition: Option<&'a Expression>,
    /// The pacing type  of the spawn, e.g. @1Hz or @input_i
    pub annotated_pacing: Option<&'a AnnotatedPacingType>,
}

impl<'a> SpawnDef<'a> {
    /// Constructs a new [SpawnDef]
    pub fn new(
        target: Option<&'a Expression>,
        condition: Option<&'a Expression>,
        annotated_pacing: Option<&'a AnnotatedPacingType>,
    ) -> Self {
        Self {
            target,
            condition,
            annotated_pacing,
        }
    }
}

/// The Hir Eval definition is composed of three expressions and the annotated pacing.
/// The first one refers to the filter expression, while the second one represents the evaluation expression, defining the value of the stream.
#[derive(Debug, Clone, Copy)]
pub struct EvalDef<'a> {
    /// The filter expression has to evaluated to true in order for the stream expression to be evaluated.
    pub filter: Option<&'a Expression>,
    /// The stream expression defines the computed value of the stream.
    pub expr: &'a Expression,
    /// The annotated pacing of the stream evaluation, describing when the filter and expression should be evaluated in a temporal manner.
    pub annotated_pacing: Option<&'a AnnotatedPacingType>,
}

impl<'a> EvalDef<'a> {
    /// Constructs a new [EvalDef]
    pub fn new(
        filter: Option<&'a Expression>,
        expr: &'a Expression,
        annotated_pacing: Option<&'a AnnotatedPacingType>,
    ) -> Self {
        Self {
            filter,
            expr,
            annotated_pacing,
        }
    }
}

/// The Hir Close definition is composed of the Close condition expression and the annotated pacing.
#[derive(Debug, Clone, Copy)]
pub struct CloseDef<'a> {
    /// The close condition, defining when a stream instance is closed and no longer evaluated.
    pub condition: Option<&'a Expression>,
    /// The annotated pacing, indicating when the condition should be evaluated.
    pub annotated_pacing: Option<&'a AnnotatedPacingType>,
}

impl<'a> CloseDef<'a> {
    /// Constructs a new [CloseDef]
    pub fn new(condition: Option<&'a Expression>, annotated_pacing: Option<&'a AnnotatedPacingType>) -> Self {
        Self {
            condition,
            annotated_pacing,
        }
    }
}

/// A [TransformationErr] describes the kind off error raised during the Ast to Hir conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationErr {
    /// A function was found when a stream was expected.
    InvalidIdentRef(Span, FunctionName),
    /// No valid streamrefernce found while it was expected.
    InvalidRefExpr(Span, String),
    /// A declared constant had no type annotation.
    ConstantWithoutType(Span),
    /// Could not parse numeric literal.
    NonNumericInLiteral(Span),
    /// Invalid activation condition.
    InvalidAc(Span, String),
    /// Offset expression of realtime offset could not be parsed as frequency.
    InvalidRealtimeOffset(Span),
    /// Window duration could not be parsed into correct type.
    InvalidDuration(String, Span),
    /// Missing expression cannot be transformed.
    MissingExpr(Span),
    /// Widen call expects a single type argument.
    MissingWidenArg(Span),
    /// Annotated type could not be matched.
    InvalidType(Type, String, Span),
    /// Called function is unknown.
    UnknownFunction(Span),
    /// Non duration literal with postfix found.
    InvalidLiteral(Span),
    /// An access to a parameterized output stream is missing its parameters
    MissingArguments(Span),
    /// An output stream with parameters is missing a spawn target
    MissingSpawn(Span),
    /// An output stream with a spawn target is missing parameters
    MissingParameters(Span),
    /// The number of parameters of an output differs from the number of spawn expressions.
    /// Span of the output, number of parameters, number of spawn expressions
    SpawnParameterMismatch(Span, usize, usize),
}

impl TransformationErr {
    pub(crate) fn into_diagnostic(self) -> Diagnostic {
        match self {
            TransformationErr::InvalidIdentRef(span, name) => {
                Diagnostic::error("Expected a stream reference, but found a function.").add_span_with_label(
                    span,
                    Some(&format!("Found function {} here", name)),
                    true,
                )
            },
            TransformationErr::InvalidRefExpr(span, reason) => {
                Diagnostic::error(&format!("Invalid stream identifier: {}", reason)).add_span_with_label(
                    span,
                    Some("Found here"),
                    true,
                )
            },
            TransformationErr::ConstantWithoutType(span) => {
                Diagnostic::error("Missing type annotation of constant stream.").add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
            },
            TransformationErr::NonNumericInLiteral(span) => {
                Diagnostic::error("Invalid numeric literal.").add_span_with_label(span, Some("here"), true)
            },
            TransformationErr::InvalidAc(span, reason) => {
                Diagnostic::error(&format!("Invalid frequency annotation: {}", reason)).add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
            },
            TransformationErr::InvalidRealtimeOffset(span) => {
                Diagnostic::error("Invalid time format.").add_span_with_label(span, Some("here"), true)
            },
            TransformationErr::InvalidDuration(reason, span) => {
                Diagnostic::error("Invalid window duration.").add_span_with_label(span, Some(&reason), true)
            },
            TransformationErr::MissingExpr(span) => {
                Diagnostic::error("Expected an expression.").add_span_with_label(span, Some("here"), true)
            },
            TransformationErr::MissingWidenArg(span) => {
                Diagnostic::error("The widen expression expects an argument.").add_span_with_label(
                    span,
                    Some("missing argument here"),
                    true,
                )
            },
            TransformationErr::InvalidType(ty, reason, span) => {
                Diagnostic::error(&format!("Unknown type {}.", ty)).add_span_with_label(span, Some(&reason), true)
            },
            TransformationErr::UnknownFunction(span) => {
                Diagnostic::error("Unknown function.").add_span_with_label(span, Some("Found here"), true)
            },
            TransformationErr::InvalidLiteral(span) => {
                Diagnostic::error("Unit declarations are not allowed on non-time literals.").add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
            },
            TransformationErr::MissingArguments(span) => {
                Diagnostic::error("An access to a parameterized output stream is missing its arguments.").add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
            }
            TransformationErr::MissingSpawn(span) => {
                Diagnostic::error("The following stream has parameters but no spawn target.").add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
                    .add_note("Help: Add a spawn declaration of the form: spawn with (exp1, ..., exp_n) if ...")
            }
            TransformationErr::MissingParameters(span) => {
                Diagnostic::error("The following stream has a spawn declaration but no parameters.").add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
            }
            TransformationErr::SpawnParameterMismatch(span, paras, targets) => {
                Diagnostic::error(&format!("The number of parameters of the stream differs from the number of spawn target expressions. Found {} parameters and {} spawn targets", paras, targets)).add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
            }
        }
    }
}

#[derive(Debug)]
struct ExpressionTransformer {
    sliding_windows: Vec<Window<SlidingAggr>>,
    discrete_windows: Vec<Window<DiscreteAggr>>,
    decl_table: HashMap<NodeId, Declaration>,
    stream_by_name: HashMap<String, SRef>,
    current_exp_id: u32,
}

impl ExpressionTransformer {
    fn run(
        decl_table: HashMap<NodeId, Declaration>,
        stream_by_name: HashMap<String, SRef>,
        ast: RtLolaAst,
        func_table: HashMap<String, FuncDecl>,
    ) -> Result<Hir<BaseMode>, TransformationErr> {
        ExpressionTransformer {
            sliding_windows: vec![],
            discrete_windows: vec![],
            decl_table,
            stream_by_name,
            current_exp_id: 0,
        }
        .transform_expressions(ast, func_table)
    }

    fn transform_expressions(
        mut self,
        ast: RtLolaAst,
        func_table: HashMap<String, FuncDecl>,
    ) -> Result<Hir<BaseMode>, TransformationErr> {
        debug_assert!(ast.mirrors.is_empty(), "Syntactic sugar should be removed beforehand.");
        let RtLolaAst {
            imports: _,   // todo
            constants: _, //handled through naming analysis
            inputs,
            outputs,
            mirrors: _,
            trigger,
            type_declarations: _,
            next_node_id: _,
        } = ast;
        let mut exprid_to_expr = HashMap::new();
        let mut hir_outputs = vec![];
        for (ix, o) in outputs.into_iter().enumerate() {
            let sr = SRef::Out(ix);
            let ast::Output {
                name,
                params,
                spawn,
                eval,
                close,
                annotated_type,
                id: _,
                span: _,
            } = (*o).clone();
            let params = params
                .iter()
                .enumerate()
                .map(|(ix, p)| {
                    assert_eq!(ix, p.param_idx);
                    p.ty.as_ref()
                        .map_or(Ok(None), |ty| {
                            Self::annotated_type(ty)
                                .map(Some)
                                .map_err(|reason| (reason, ty.clone(), p.span.clone()))
                        })
                        .map(|p_ty| {
                            Parameter {
                                name: p.name.name.clone(),
                                annotated_type: p_ty,
                                idx: p.param_idx,
                                span: p.span.clone(),
                            }
                        })
                })
                .collect::<Result<Vec<Parameter>, (String, Type, Span)>>()
                .map_err(|(reason, ty, span)| TransformationErr::InvalidType(ty, reason, span))?;
            let annotated_type = annotated_type
                .as_ref()
                .map_or(Ok(None), |ty| {
                    Self::annotated_type(ty)
                        .map(Some)
                        .map_err(|reason| (reason, ty.clone(), ty.span.clone()))
                })
                .map_err(|(reason, ty, span)| TransformationErr::InvalidType(ty, reason, span))?;

            let spawn = self.transform_spawn_clause(spawn, &mut exprid_to_expr, sr)?;
            let eval = self.transform_eval_clause(eval, &mut exprid_to_expr, sr)?;
            let close = self.transform_close_clause(close, &mut exprid_to_expr, sr)?;

            //Check that if the output has parameters it has a spawn condition with a target and the other way around
            if !params.is_empty() && spawn.as_ref().and_then(|st| st.target).is_none() {
                return Err(TransformationErr::MissingSpawn(o.span.clone()));
            }

            if let Some(target) = spawn.as_ref().and_then(|st| st.target) {
                let spawn_expr = &exprid_to_expr[&target];
                if params.is_empty() {
                    return Err(TransformationErr::MissingParameters(spawn_expr.span.clone()));
                }
                // check that they are equal length
                let num_spawn_expr = match &spawn_expr.kind {
                    ExpressionKind::Tuple(elements) => elements.len(),
                    _ => 1,
                };
                if num_spawn_expr != params.len() {
                    return Err(TransformationErr::SpawnParameterMismatch(
                        o.span.clone(),
                        params.len(),
                        num_spawn_expr,
                    ));
                }
            }

            hir_outputs.push(Output {
                name: name.name,
                sr,
                params,
                spawn,
                eval,
                close,
                annotated_type,
                span: o.span.clone(),
            });
        }
        let hir_outputs = hir_outputs;
        let mut hir_triggers = vec![];
        for (ix, t) in trigger.into_iter().enumerate() {
            let sr = SRef::Out(hir_outputs.len() + ix);
            let ast::Trigger {
                message,
                annotated_pacing_type,
                info_streams,
                expression,
                span,
                ..
            } = Rc::try_unwrap(t).expect("other strong references should be dropped now");
            let pt = annotated_pacing_type
                .map_or(Ok(None), |pt| self.transform_pt(&mut exprid_to_expr, pt, sr).map(Some))?;
            let info_streams: Vec<SRef> = info_streams
                .into_iter()
                .map(|ident| {
                    *self
                        .stream_by_name
                        .get(&ident.name)
                        .expect("Ensured by naming analysis")
                })
                .collect();
            let expr_id = Self::insert_return(&mut exprid_to_expr, self.transform_expression(expression, sr)?);

            hir_triggers.push(Trigger::new(message, info_streams, pt, expr_id, sr, span));
        }
        let hir_triggers = hir_triggers;
        let hir_inputs: Vec<Input> = inputs
            .into_iter()
            .enumerate()
            .map(|(ix, i)| {
                Ok(Input {
                    annotated_type: Self::annotated_type(&i.ty)
                        .map_err(|reason| TransformationErr::InvalidType(i.ty.clone(), reason, i.span.clone()))?,
                    name: i.name.name.clone(),
                    sr: SRef::In(ix),
                    span: i.span.clone(),
                })
            })
            .collect::<Result<Vec<_>, TransformationErr>>()?;

        let ExpressionTransformer {
            sliding_windows,
            discrete_windows,
            ..
        } = self;
        let sliding_windows = sliding_windows.into_iter().map(|w| (w.reference, w)).collect();
        let discrete_windows = discrete_windows.into_iter().map(|w| (w.reference, w)).collect();
        let expr_maps = ExpressionMaps::new(exprid_to_expr, sliding_windows, discrete_windows, func_table);

        let new_mode = BaseMode {};

        Ok(Hir {
            next_input_ref: hir_inputs.len(),
            inputs: hir_inputs,
            next_output_ref: hir_outputs.len(),
            outputs: hir_outputs,
            triggers: hir_triggers,
            expr_maps,
            mode: new_mode,
        })
    }

    fn annotated_type(ast_ty: &Type) -> Result<AnnotatedType, String> {
        use rtlola_parser::ast::TypeKind;
        match &ast_ty.kind {
            TypeKind::Tuple(vec) => {
                let inner: Result<Vec<AnnotatedType>, String> = vec.iter().map(Self::annotated_type).collect();
                inner.map(AnnotatedType::Tuple)
            },
            TypeKind::Optional(inner) => Self::annotated_type(inner).map(|inner| AnnotatedType::Option(inner.into())),
            TypeKind::Simple(string) => {
                if string == "String" {
                    return Ok(AnnotatedType::String);
                }
                if string == "Bool" {
                    return Ok(AnnotatedType::Bool);
                }
                if let Some(size_str) = string.strip_prefix("Int") {
                    if string.len() == 3 {
                        return Ok(AnnotatedType::Int(64));
                    } else {
                        let size: u32 = size_str
                            .parse()
                            .map_err(|_| "Invalid char followed Int type annotation".to_string())?;
                        return Ok(AnnotatedType::Int(size));
                    }
                }
                if let Some(size_str) = string.strip_prefix("UInt") {
                    if string.len() == 4 {
                        return Ok(AnnotatedType::UInt(64));
                    } else {
                        let size: u32 = size_str
                            .parse()
                            .map_err(|_| "Invalid char followed UInt type annotation".to_string())?;
                        return Ok(AnnotatedType::UInt(size));
                    }
                }
                if let Some(size_str) = string.strip_prefix("Float") {
                    if string.len() == 5 {
                        return Ok(AnnotatedType::Float(64));
                    } else {
                        let size: u32 = size_str
                            .parse()
                            .map_err(|_| "Invalid char followed Float type annotation".to_string())?;
                        return Ok(AnnotatedType::Float(size));
                    }
                }
                if string == "Bytes" {
                    return Ok(AnnotatedType::Bytes);
                }
                Err("unknown type".into())
            },
        }
    }

    fn get_stream_ref(
        &mut self,
        expr: &ast::Expression,
        current_output: SRef,
    ) -> Result<(SRef, Vec<Expression>), TransformationErr> {
        match &expr.kind {
            ast::ExpressionKind::Ident(_) => {
                match &self.decl_table[&expr.id] {
                    Declaration::In(i) => Ok((self.stream_by_name[&i.name.name], Vec::new())),
                    Declaration::Out(o) => Ok((self.stream_by_name[&o.name.name], Vec::new())),
                    Declaration::ParamOut(_) => Err(TransformationErr::MissingArguments(expr.span.clone())),
                    _ => {
                        Err(TransformationErr::InvalidRefExpr(
                            expr.span.clone(),
                            String::from("Non-identifier transformed to SRef"),
                        ))
                    },
                }
            },
            ast::ExpressionKind::Function(name, _, args) => {
                match &self.decl_table[&expr.id] {
                    Declaration::ParamOut(o) => {
                        Ok((
                            self.stream_by_name[&o.name.name],
                            args.iter()
                                .map(|e| self.transform_expression(e.clone(), current_output))
                                .collect::<Result<Vec<_>, TransformationErr>>()?,
                        ))
                    },
                    _ => Err(TransformationErr::InvalidIdentRef(expr.span.clone(), name.clone())),
                }
            },
            _ => {
                Err(TransformationErr::InvalidRefExpr(
                    expr.span.clone(),
                    format!("{:?}", expr.kind),
                ))
            },
        }
    }

    fn next_exp_id(&mut self) -> ExprId {
        let ret = self.current_exp_id;
        self.current_exp_id += 1;
        ExprId(ret)
    }

    fn transform_literal(&self, lit: &AstLiteral) -> Result<Literal, TransformationErr> {
        Ok(match &lit.kind {
            ast::LitKind::Bool(b) => Literal::Bool(*b),
            ast::LitKind::Str(s) | ast::LitKind::RawStr(s) => Literal::Str(s.clone()),
            ast::LitKind::Numeric(num_str, postfix) => {
                match postfix {
                    Some(s) if !s.is_empty() => return Err(TransformationErr::InvalidLiteral(lit.span.clone())),
                    _ => {},
                }

                if num_str.contains('.') {
                    // Floating Point
                    Literal::Float(
                        num_str
                            .parse()
                            .map_err(|_| TransformationErr::NonNumericInLiteral(lit.span.clone()))?,
                    )
                } else if num_str.starts_with('-') {
                    Literal::SInt(
                        num_str
                            .parse()
                            .map_err(|_| TransformationErr::NonNumericInLiteral(lit.span.clone()))?,
                    )
                } else {
                    Literal::Integer(
                        num_str
                            .parse()
                            .map_err(|_| TransformationErr::NonNumericInLiteral(lit.span.clone()))?,
                    )
                }
            },
        })
    }

    fn transform_pt(
        &mut self,
        exprid_to_expr: &mut HashMap<ExprId, Expression>,
        pt_expr: ast::Expression,
        current: SRef,
    ) -> Result<AnnotatedPacingType, TransformationErr> {
        if let ast::ExpressionKind::Lit(l) = &pt_expr.kind {
            if let ast::LitKind::Numeric(_, Some(_)) = &l.kind {
                let val = pt_expr
                    .parse_freqspec()
                    .map_err(|reason| TransformationErr::InvalidAc(pt_expr.span.clone(), reason))?;
                return Ok(AnnotatedPacingType::Frequency {
                    span: pt_expr.span.clone(),
                    value: val,
                });
            }
        }
        Ok(AnnotatedPacingType::Expr(Self::insert_return(
            exprid_to_expr,
            self.transform_expression(pt_expr, current)?,
        )))
    }

    fn transform_expression(
        &mut self,
        ast_expression: ast::Expression,
        current_output: SRef,
    ) -> Result<Expression, TransformationErr> {
        let new_id = self.next_exp_id();
        let span = ast_expression.span;
        let kind: ExpressionKind = match ast_expression.kind {
            ast::ExpressionKind::Lit(lit) => {
                let constant = self.transform_literal(&lit)?;
                ExpressionKind::LoadConstant(HirConstant::Basic(constant))
            },
            ast::ExpressionKind::Ident(_) => {
                match &self.decl_table[&ast_expression.id] {
                    Declaration::Out(o) => {
                        let sr = self.stream_by_name[&o.name.name];
                        ExpressionKind::StreamAccess(sr, IRAccess::Sync, Vec::new())
                    },
                    Declaration::In(i) => {
                        let sr = self.stream_by_name[&i.name.name];
                        ExpressionKind::StreamAccess(sr, IRAccess::Sync, Vec::new())
                    },
                    Declaration::Const(c) => {
                        let ty =
                            c.ty.as_ref()
                                .ok_or_else(|| TransformationErr::ConstantWithoutType(span.clone()))?;
                        let annotated_type = Self::annotated_type(ty)
                            .map_err(|reason| TransformationErr::InvalidType(ty.clone(), reason, span.clone()))?;
                        ExpressionKind::LoadConstant(HirConstant::Inlined(Inlined {
                            lit: self.transform_literal(&c.literal)?,
                            ty: annotated_type,
                        }))
                    },

                    Declaration::Param(p) => ExpressionKind::ParameterAccess(current_output, p.param_idx),
                    Declaration::ParamOut(_) => {
                        return Err(TransformationErr::MissingArguments(span));
                    },
                    Declaration::Func(_) | Declaration::Type(_) => {
                        unreachable!("Identifiers can only refer to streams")
                    },
                }
            },
            ast::ExpressionKind::StreamAccess(expr, kind) => {
                let access_kind = if let StreamAccessKind::Hold = kind {
                    IRAccess::Hold
                } else {
                    IRAccess::Sync
                };
                let (expr_ref, args) = self.get_stream_ref(&*expr, current_output)?;
                ExpressionKind::StreamAccess(expr_ref, access_kind, args)
            },
            ast::ExpressionKind::Default(expr, def) => {
                ExpressionKind::Default {
                    expr: Box::new(self.transform_expression(*expr, current_output)?),
                    default: Box::new(self.transform_expression(*def, current_output)?),
                }
            },
            ast::ExpressionKind::Offset(ref target_expr, offset) => {
                use uom::si::time::nanosecond;
                let ir_offset = match offset {
                    ast::Offset::Discrete(i) if i == 0 => None,
                    ast::Offset::Discrete(i) if i > 0 => Some(Offset::FutureDiscrete(i.unsigned_abs().into())),
                    ast::Offset::Discrete(i) => Some(Offset::PastDiscrete(i.unsigned_abs().into())),
                    ast::Offset::RealTime(_, _) => {
                        let offset_uom_time = offset
                            .to_uom_time()
                            .ok_or_else(|| TransformationErr::InvalidRealtimeOffset(span.clone()))?;
                        let dur = offset_uom_time.get::<nanosecond>().to_integer();
                        //TODO FIXME check potential loss of precision
                        let time = offset_uom_time.get::<nanosecond>();
                        let numer = time.numer();
                        match numer {
                            0 => None,
                            i if i < &0 => {
                                let positive_dur = Duration::from_nanos((-dur) as u64);
                                Some(Offset::PastRealTime(positive_dur))
                            },
                            _ => {
                                let positive_dur = Duration::from_nanos(dur as u64);
                                Some(Offset::FutureRealTime(positive_dur))
                            },
                        }
                    },
                };
                let (expr_ref, args) = self.get_stream_ref(&*target_expr, current_output)?;
                let kind = ir_offset.map(IRAccess::Offset).unwrap_or(IRAccess::Sync);
                ExpressionKind::StreamAccess(expr_ref, kind, args)
            },
            ast::ExpressionKind::DiscreteWindowAggregation {
                expr: w_expr,
                duration,
                wait,
                aggregation: win_op,
            } => {
                let (sref, paras) = self.get_stream_ref(&w_expr, current_output)?;
                let idx = self.discrete_windows.len();
                let wref = WRef::Discrete(idx);
                let duration = (*duration)
                    .parse_discrete_duration()
                    .map_err(|e| TransformationErr::InvalidDuration(e, span.clone()))?;
                let window = Window {
                    target: sref,
                    caller: current_output,
                    aggr: DiscreteAggr {
                        wait,
                        op: win_op,
                        duration: duration.try_into().unwrap(),
                    },
                    reference: wref,
                    eid: new_id,
                };
                self.discrete_windows.push(window);
                ExpressionKind::StreamAccess(sref, IRAccess::DiscreteWindow(WRef::Discrete(idx)), paras)
            },
            ast::ExpressionKind::SlidingWindowAggregation {
                expr: w_expr,
                duration,
                wait,
                aggregation: win_op,
            } => {
                let (sref, paras) = self.get_stream_ref(&w_expr, current_output)?;
                let idx = self.sliding_windows.len();
                let wref = WRef::Sliding(idx);
                let duration = Self::parse_duration_from_expr(&*duration)
                    .map_err(|e| TransformationErr::InvalidDuration(e, span.clone()))?;
                let window = Window {
                    target: sref,
                    caller: current_output,
                    aggr: SlidingAggr {
                        duration,
                        wait,
                        op: win_op,
                    },
                    reference: wref,
                    eid: new_id,
                };
                self.sliding_windows.push(window);
                ExpressionKind::StreamAccess(sref, IRAccess::SlidingWindow(WRef::Sliding(idx)), paras)
            },
            ast::ExpressionKind::Binary(op, left, right) => {
                use rtlola_parser::ast::BinOp;

                use crate::hir::ArithLogOp::*;
                let arith_op = match op {
                    BinOp::Add => Add,
                    BinOp::Sub => Sub,
                    BinOp::Mul => Mul,
                    BinOp::Div => Div,
                    BinOp::Rem => Rem,
                    BinOp::Pow => Pow,
                    BinOp::And => And,
                    BinOp::Or => Or,
                    BinOp::BitXor => BitXor,
                    BinOp::BitAnd => BitAnd,
                    BinOp::BitOr => BitOr,
                    BinOp::Shl => Shl,
                    BinOp::Shr => Shr,
                    BinOp::Eq => Eq,
                    BinOp::Lt => Lt,
                    BinOp::Le => Le,
                    BinOp::Ne => Ne,
                    BinOp::Ge => Ge,
                    BinOp::Gt => Gt,
                };
                let arguments: Vec<Expression> = vec![
                    self.transform_expression(*left, current_output)?,
                    self.transform_expression(*right, current_output)?,
                ];
                ExpressionKind::ArithLog(arith_op, arguments)
            },
            ast::ExpressionKind::Unary(op, arg) => {
                use rtlola_parser::ast::UnOp;

                use crate::hir::ArithLogOp::*;
                let arith_op = match op {
                    UnOp::Not => Not,
                    UnOp::Neg => Neg,
                    UnOp::BitNot => BitNot,
                };
                let arguments: Vec<Expression> = vec![self.transform_expression(*arg, current_output)?];
                ExpressionKind::ArithLog(arith_op, arguments)
            },
            ast::ExpressionKind::Ite(cond, cons, alt) => {
                let condition = Box::new(self.transform_expression(*cond, current_output)?);
                let consequence = Box::new(self.transform_expression(*cons, current_output)?);
                let alternative = Box::new(self.transform_expression(*alt, current_output)?);
                ExpressionKind::Ite {
                    condition,
                    consequence,
                    alternative,
                }
            },
            ast::ExpressionKind::ParenthesizedExpression(_, inner, _) => {
                return self.transform_expression(*inner, current_output);
            },
            ast::ExpressionKind::MissingExpression => return Err(TransformationErr::MissingExpr(span)),
            ast::ExpressionKind::Tuple(inner) => {
                ExpressionKind::Tuple(
                    inner
                        .into_iter()
                        .map(|ex| self.transform_expression(ex, current_output))
                        .collect::<Result<Vec<_>, TransformationErr>>()?,
                )
            },
            ast::ExpressionKind::Field(inner_exp, ident) => {
                let num: usize = ident.name.parse().expect("checked in AST verifier");
                let inner = Box::new(self.transform_expression(*inner_exp, current_output)?);
                ExpressionKind::TupleAccess(inner, num)
            },
            ast::ExpressionKind::Method(base, name, type_param, mut args) => {
                // Method Access is same as function application with base as first parameter
                args.insert(0, *base);
                self.transfrom_function(
                    false,
                    ast_expression.id,
                    &span,
                    current_output,
                    ast::ExpressionKind::Function(name, type_param, args),
                )?
            },
            ast::ExpressionKind::Function(..) => {
                self.transfrom_function(true, ast_expression.id, &span, current_output, ast_expression.kind)?
            },
        };
        Ok(Expression {
            kind,
            eid: new_id,
            span,
        })
    }

    /// Unifies the transformation of function and method applications to the internal representation
    fn transfrom_function(
        &mut self,
        allow_parametric: bool,
        id: NodeId,
        span: &Span,
        current_output: SRef,
        kind: ast::ExpressionKind,
    ) -> Result<ExpressionKind, TransformationErr> {
        let (name, type_param, args) = if let ast::ExpressionKind::Function(name, type_param, args) = kind {
            (name, type_param, args)
        } else {
            unreachable!()
        };
        let decl = self
            .decl_table
            .get(&id)
            .ok_or_else(|| TransformationErr::UnknownFunction(span.clone()))?;
        match decl {
            Declaration::Func(_) => {
                let name = name.name.name;
                let args: Vec<Expression> = args
                    .into_iter()
                    .map(|ex| self.transform_expression(ex, current_output))
                    .collect::<Result<Vec<_>, TransformationErr>>()?;

                if name.starts_with("widen") {
                    let widen_arg = args
                        .get(0)
                        .ok_or_else(|| TransformationErr::MissingWidenArg(span.clone()))?;
                    Ok(ExpressionKind::Widen(WidenExprKind {
                        expr: Box::new(widen_arg.clone()),
                        ty: match type_param.get(0) {
                            Some(t) => {
                                Self::annotated_type(t)
                                    .map_err(|reason| TransformationErr::InvalidType(t.clone(), reason, span.clone()))?
                            },
                            None => todo!("error case"),
                        },
                    }))
                } else {
                    Ok(ExpressionKind::Function(FnExprKind {
                        name,
                        args,
                        type_param: type_param
                            .into_iter()
                            .map(|t| {
                                Self::annotated_type(&t)
                                    .map_err(|reason| TransformationErr::InvalidType(t, reason, span.clone()))
                            })
                            .collect::<Result<Vec<_>, TransformationErr>>()?,
                    }))
                }
            },
            Declaration::ParamOut(_) => {
                if allow_parametric {
                    Ok(ExpressionKind::StreamAccess(
                        self.stream_by_name[&name.name.name],
                        IRAccess::Sync,
                        args.into_iter()
                            .map(|ex| self.transform_expression(ex, current_output))
                            .collect::<Result<Vec<_>, TransformationErr>>()?,
                    ))
                } else {
                    Err(TransformationErr::UnknownFunction(span.clone()))
                }
            },
            _ => Err(TransformationErr::UnknownFunction(span.clone())),
        }
    }

    /// Converts an expression into the internal representation for a duration
    /// e.g. parse_duration_from_expr(5Hz) == Duration::from_nanos(200)
    fn parse_duration_from_expr(ast_expression: &ast::Expression) -> Result<Duration, String> {
        use num::traits::Inv;
        use num::ToPrimitive;
        use uom::si::frequency::hertz;
        use uom::si::rational64::Time as UOM_Time;
        use uom::si::time::second;

        let freq = ast_expression.parse_freqspec()?;
        let period = UOM_Time::new::<second>(freq.get::<hertz>().inv());
        let nanos = period
            .get::<uom::si::time::nanosecond>()
            .to_integer()
            .to_u64()
            .ok_or_else(|| String::from("Period to large to fit into u64"))?;
        Ok(Duration::from_nanos(nanos))
    }

    /// Adds an expression Id and the expression into the hash map and returns the id.
    fn insert_return(exprid_to_expr: &mut HashMap<ExprId, Expression>, expr: Expression) -> ExprId {
        let id = expr.eid;
        exprid_to_expr.insert(id, expr);
        id
    }

    fn transform_spawn_clause(
        &mut self,
        spawn_spec: Option<SpawnSpec>,
        exprid_to_expr: &mut HashMap<ExprId, Expression>,
        current_output: SRef,
    ) -> Result<Option<SpawnTemplate>, TransformationErr> {
        spawn_spec.map_or(Ok(None), |spawn_spec| {
            let SpawnSpec {
                target,
                annotated_pacing,
                condition,
                ..
            } = spawn_spec;
            let target = target.map_or(Ok(None), |target_exp| {
                if let ast::ExpressionKind::ParenthesizedExpression(_, ref exp, _) = target_exp.kind {
                    if let ast::ExpressionKind::MissingExpression = exp.kind {
                        return Ok(None);
                    }
                }
                let exp = self.transform_expression(target_exp, current_output)?;
                Ok(Some(Self::insert_return(exprid_to_expr, exp)))
            })?;
            let pacing = annotated_pacing.map_or(Ok(None), |pt| {
                Ok(Some(self.transform_pt(exprid_to_expr, pt, current_output)?))
            })?;

            let condition = condition.map_or(Ok(None), |cond_expr| {
                let e = self.transform_expression(cond_expr, current_output)?;
                Ok(Some(Self::insert_return(exprid_to_expr, e)))
            })?;
            Ok(Some(SpawnTemplate {
                target,
                pacing,
                condition,
            }))
        })
    }

    fn transform_eval_clause(
        &mut self,
        eval_spec: Vec<ast::EvalSpec>,
        exprid_to_expr: &mut HashMap<ExprId, Expression>,
        current_output: SRef,
    ) -> Result<EvalTemplate, TransformationErr> {
        assert!(eval_spec.len() <= 1); //TODO remove in upcoming refactoring
        let evt = if let Some(eval_spec) = eval_spec.get(0) {
            let eval_spec = eval_spec.to_owned();
            let eval_expr = if let Some(eval_expr) = eval_spec.eval_expression {
                self.transform_expression(eval_expr, current_output)?
            } else {
                unreachable!("Empty tuple is inserted if the expression is unspecified or the parser reports an error");
            };
            let eval_expr_id = Self::insert_return(exprid_to_expr, eval_expr);

            let filter_expr = eval_spec.filter.map_or(Ok(None), |fe| {
                let filter_expr = self.transform_expression(fe, current_output)?;
                Ok(Some(Self::insert_return(exprid_to_expr, filter_expr)))
            })?;

            let pacing = eval_spec.annotated_pacing.map_or(Ok(None), |pt| {
                Ok(Some(self.transform_pt(exprid_to_expr, pt, current_output)?))
            })?;

            EvalTemplate {
                expr: eval_expr_id,
                filter: filter_expr,
                annotated_pacing_type: pacing,
            }
        } else {
            unreachable!("Grammar requires at least one eval declaration")
        };
        Ok(evt)
    }

    fn transform_close_clause(
        &mut self,
        close_spec: Option<ast::CloseSpec>,
        exprid_to_expr: &mut HashMap<ExprId, Expression>,
        current_output: SRef,
    ) -> Result<Option<CloseTemplate>, TransformationErr> {
        close_spec.map_or(Ok(None), |close_spec| {
            let pacing = close_spec.annotated_pacing.map_or(Ok(None), |pt| {
                Ok(Some(self.transform_pt(exprid_to_expr, pt, current_output)?))
            })?;
            let target = Self::insert_return(
                exprid_to_expr,
                self.transform_expression(close_spec.target, current_output)?,
            );
            Ok(Some(CloseTemplate {
                condition: target,
                pacing,
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rtlola_parser::ast::WindowOperation;
    use rtlola_parser::{parse, ParserConfig};

    use super::*;
    use crate::hir::{ExpressionContext, StreamAccessKind};

    fn obtain_expressions(spec: &str) -> Hir<BaseMode> {
        let ast = parse(ParserConfig::for_string(spec.to_string())).unwrap_or_else(|e| panic!("{:?}", e));
        crate::from_ast(ast).unwrap()
    }

    #[test]
    fn all() {
        //Tests all cases are implemented
        let spec = "
        import math\n\
        input i: Int8\n\
        output o := 3\n\
        output o2 @1Hz := 4\n\
        output o3 := if true then 1 else 2\n\
        output o4 := 3 + 4\n\
        output o5 := o\n\
        output f := sqrt(4)\n\
        output p (x,y) spawn when true with (i,i) eval with x\n\
        output t := (1,3)\n\
        output t1 := t.0\n\
        output def := o.defaults(to:2)\n\
        output off := o.offset(by:-1)\n\
        output w := o2.aggregate(over:3s , using: sum)\n";
        let _ir = obtain_expressions(spec);
        //TODO
    }

    #[test]
    fn transform_default() {
        let spec = "input o :Int8 output off := o.defaults(to:-1)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].eval_expr();
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(expr.kind, ExpressionKind::Default { .. }));
    }

    #[test]
    fn transform_offset() {
        use crate::hir::StreamAccessKind;
        //TODO do remaining cases
        for (spec, offset) in &[
            ("input o :Int8 output off := o", StreamAccessKind::Sync),
            //("input o :Int8 output off := o.aggregate(over: 1s, using: sum)",StreamAccessKind::DiscreteWindow(WRef::SlidingRef(0))),
            ("input o :Int8 output off := o.hold()", StreamAccessKind::Hold),
            (
                "input o :Int8 output off := o.offset(by:-1)",
                StreamAccessKind::Offset(Offset::PastDiscrete(1)),
            ),
            (
                "input o :Int8 output off := o.offset(by: 1)",
                StreamAccessKind::Offset(Offset::FutureDiscrete(1)),
            ),
            ("input o :Int8 output off := o.offset(by: 0)", StreamAccessKind::Sync),
            (
                "input o :Int8 output off := o.offset(by:-1s)",
                StreamAccessKind::Offset(Offset::PastRealTime(Duration::from_secs(1))),
            ),
            (
                "input o :Int8 output off := o.offset(by: 1s)",
                StreamAccessKind::Offset(Offset::FutureRealTime(Duration::from_secs(1))),
            ),
            ("input o :Int8 output off := o.offset(by: 0s)", StreamAccessKind::Sync),
        ] {
            let ir = obtain_expressions(spec);
            let output_expr_id = ir.outputs[0].eval().expr;
            let expr = &ir.expression(output_expr_id);
            assert!(matches!(expr.kind, ExpressionKind::StreamAccess(SRef::In(0), _, _)));
            if let ExpressionKind::StreamAccess(SRef::In(0), result_kind, _) = expr.kind {
                assert_eq!(result_kind, *offset);
            }
        }
    }

    #[test]
    fn transform_aggr() {
        use std::time::Duration;

        use crate::hir::SlidingAggr;
        let spec = "input i:Int8 output o := i.aggregate(over: 1s, using: sum)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].eval_expr();
        let expr = &ir.expression(output_expr_id);
        let wref = WRef::Sliding(0);
        assert!(matches!(
            expr.kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::SlidingWindow(WRef::Sliding(0)), _)
        ));
        let window = ir.single_sliding(WRef::Sliding(0)).clone();
        let aggr = SlidingAggr {
            wait: false,
            op: WindowOperation::Sum,
            duration: Duration::from_secs(1),
        };
        assert_eq!(
            window,
            Window {
                target: SRef::In(0),
                caller: SRef::Out(0),
                aggr,
                reference: wref,
                eid: ExprId(0),
            }
        );
    }

    #[test]
    fn parameter_expr() {
        let spec = "output o(a,b,c) spawn @1Hz with (1, 2, 3) eval with if c then a else b";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].eval().expr;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(expr.kind, ExpressionKind::Ite { .. }));
        if let ExpressionKind::Ite {
            condition,
            consequence,
            alternative,
        } = &expr.kind
        {
            assert!(matches!(condition.kind, ExpressionKind::ParameterAccess(_, 2)));
            assert!(matches!(consequence.kind, ExpressionKind::ParameterAccess(_, 0)));
            assert!(matches!(alternative.kind, ExpressionKind::ParameterAccess(_, 1)));
        } else {
            unreachable!()
        }
    }

    #[test]
    fn parametrized_access() {
        use crate::hir::StreamAccessKind;
        let spec =
            "output o(a,b,c) spawn @1Hz with (1, 2, true) eval with if c then a else b output A := o(1,2,true).offset(by:-1)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[1].eval_expr();
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(
            expr.kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Offset(Offset::PastDiscrete(_)), _)
        ));
        if let ExpressionKind::StreamAccess(sr, _, v) = &expr.kind {
            assert_eq!(*sr, SRef::Out(0));
            assert_eq!(v.len(), 3);
        } else {
            unreachable!()
        }
    }

    #[test]
    fn tuple() {
        let spec = "output o := (1,2,3)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].eval().expr;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(expr.kind, ExpressionKind::Tuple(_)));
        if let ExpressionKind::Tuple(v) = &expr.kind {
            assert_eq!(v.len(), 3);
            for atom in v.iter() {
                assert!(matches!(atom.kind, ExpressionKind::LoadConstant(_)));
            }
        } else {
            unreachable!()
        }
    }

    #[test]
    fn tuple_access() {
        let spec = "output o := (1,2,3).1";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].eval().expr;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(expr.kind, ExpressionKind::TupleAccess(_, 1)));
    }

    #[test]
    fn simple_trigger() {
        let spec = "trigger true";
        let ir = obtain_expressions(spec);
        assert_eq!(ir.num_triggers(), 1);
        let tr = &ir.triggers[0];
        let expr = ir.eval_unchecked(tr.sr).expr;
        assert!(matches!(expr.kind, ExpressionKind::LoadConstant(_)));
    }

    #[test]
    fn input_trigger() {
        use crate::hir::ArithLogOp;
        let spec = "input a: Int8\n trigger a == 42";
        let ir = obtain_expressions(spec);
        assert_eq!(ir.num_triggers(), 1);
        let tr = &ir.triggers[0];
        //let expr = &ir.mode.exprid_to_expr[&tr.expr_id];
        let expr = ir.eval_unchecked(tr.sr).expr;
        assert!(matches!(expr.kind, ExpressionKind::ArithLog(ArithLogOp::Eq, _)));
    }

    #[test]
    fn arith_op() {
        use crate::hir::ArithLogOp;
        let spec = "output o := 3 + 5 ";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].eval().expr;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(expr.kind, ExpressionKind::ArithLog(ArithLogOp::Add, _)));
        if let ExpressionKind::ArithLog(_, v) = &expr.kind {
            assert_eq!(v.len(), 2);
            for atom in v.iter() {
                assert!(matches!(atom.kind, ExpressionKind::LoadConstant(_)));
            }
        } else {
            unreachable!()
        }
    }

    #[test]
    fn type_annotations() {
        for (spec, ty) in &[
            ("input i :String", AnnotatedType::String),
            ("input i :Bytes", AnnotatedType::Bytes),
            ("input i :Bool", AnnotatedType::Bool),
            (
                "input i :(Int8,UInt8)",
                AnnotatedType::Tuple(vec![AnnotatedType::Int(8), AnnotatedType::UInt(8)]),
            ),
            (
                "input i :Float32?",
                AnnotatedType::Option(Box::new(AnnotatedType::Float(32))),
            ),
        ] {
            let ir = obtain_expressions(spec);
            let input = &ir.inputs[0];
            let transformed_type = &input.annotated_type;
            assert_eq!(transformed_type, ty);
        }
    }

    #[test]
    fn functions() {
        use crate::hir::StreamAccessKind;
        let spec = "import math output o(a: Int) spawn @1Hz with 3 eval with max(3,4) output c := o(1)";
        let ir = obtain_expressions(spec);
        //check that this functions exists
        let _ = ir.func_declaration("max");
        let output_expr_id = ir.outputs[1].eval().expr;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(
            expr.kind,
            ExpressionKind::StreamAccess(SRef::Out(0), StreamAccessKind::Sync, _)
        ));
    }

    #[test]
    fn function_param_default() {
        let spec = "import math output o(a: Int) spawn @1Hz with 3 eval with sqrt(a) output c := o(1).defaults(to:1)";
        let ir = obtain_expressions(spec);
        //check purely for valid access
        let _ = ir.func_declaration("sqrt");
        let output_expr_id = ir.outputs[1].eval().expr;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(expr.kind, ExpressionKind::Default { expr: _, default: _ }));
        if let ExpressionKind::Default { expr: ex, default } = &expr.kind {
            assert!(matches!(default.kind, ExpressionKind::LoadConstant(_)));
            assert!(matches!(ex.kind, ExpressionKind::StreamAccess(SRef::Out(0), _, _)));
        } else {
            unreachable!()
        }
    }

    #[test]
    fn test_instance() {
        use crate::hir::{AnnotatedPacingType, Output};
        let spec = "input i: Bool output c(a: Int8): Bool spawn when i with i close when i eval @1Hz when i with i";
        let ir = obtain_expressions(spec);
        let output: Output = ir.outputs[0].clone();
        assert!(output.annotated_type.is_some());
        assert!(matches!(
            output.eval_pacing(),
            Some(AnnotatedPacingType::Frequency { .. })
        ));
        assert!(output.params.len() == 1);
        let fil: &Expression = ir.eval_unchecked(output.sr).filter.unwrap();
        assert!(matches!(
            fil.kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
        let close = ir.close_unchecked(output.sr).condition;
        assert!(matches!(
            close.unwrap().kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
        let SpawnDef { target, condition, .. } = ir.spawn_unchecked(output.sr);
        assert!(matches!(
            target.unwrap().kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
        assert!(matches!(
            condition.unwrap().kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
    }

    #[test]
    fn test_ac() {
        use crate::hir::{AnnotatedPacingType, Output};
        let spec = "input in: Bool output a: Bool @2.5Hz := true output b: Bool @in := false";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(matches!(a.eval_pacing(), Some(AnnotatedPacingType::Frequency { .. })));
        let b: &Output = &ir.outputs[1];
        assert!(matches!(b.eval_pacing(), Some(AnnotatedPacingType::Expr(_))));
    }

    #[test]
    fn test_spawn_missing_exp() {
        use crate::hir::{Output, SpawnTemplate};
        let spec = "output a: Bool spawn when true with () eval @2.5Hz with true";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(matches!(a.spawn().unwrap(), SpawnTemplate { target: None, .. }));
    }

    #[test]
    fn test_spawn_pacing() {
        use crate::hir::Output;
        let spec = "output a: Bool spawn @1Hz when true with () eval @2.5Hz with true";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(a.spawn().is_some());
        assert!(matches!(a.spawn_pacing(), Some(AnnotatedPacingType::Frequency { .. })));
    }

    #[test]
    fn test_trigger_info() {
        let spec = "input a:Bool\noutput b:Int8 := 42\n trigger a \"a is true\" (a, b)";
        let ir = obtain_expressions(spec);
        let trigger: Trigger = ir.triggers[0].clone();
        assert_eq!(trigger.info_streams[0], SRef::In(0));
        assert_eq!(trigger.info_streams[1], SRef::Out(0));
    }

    #[test]
    fn test_missing_eval_with_filter() {
        let spec = "input a:Bool\noutput b: Int8 eval when a > 3";
        let ir = obtain_expressions(spec);
        let out = ir.outputs[0].clone();
        assert!(out.eval_filter().is_some());
        assert!(matches!(ir.expression(out.eval_expr()).kind, ExpressionKind::Tuple(_),));
    }

    #[test]
    fn test_trigger_ac() {
        let spec = "input a:Bool\n trigger @1Hz a";
        let ir = obtain_expressions(spec);
        let trigger: Trigger = ir.triggers[0].clone();
        assert!(matches!(
            trigger.annotated_pacing_type,
            Some(AnnotatedPacingType::Frequency { .. })
        ))
    }

    #[test]
    fn test_trigger_complex() {
        let spec = "input a:Bool\n trigger @1Hz a \"This is a message\" (a)";
        let ir = obtain_expressions(spec);
        let trigger: Trigger = ir.triggers[0].clone();
        assert_eq!(trigger.info_streams[0], SRef::In(0));
        assert!(matches!(
            trigger.annotated_pacing_type,
            Some(AnnotatedPacingType::Frequency { .. })
        ))
    }

    #[test]
    fn test_instance_window() {
        let spec = "input a: Int32\n\
        output b (p: Bool) spawn with a = 42 eval with a\n\
        output c @ 1Hz := b(false).aggregate(over: 1s, using: Σ)\n";
        let ir = obtain_expressions(spec);
        let expr = &ir.expression(ir.outputs[1].eval_expr()).kind;
        assert!(
            matches!(expr, ExpressionKind::StreamAccess(_, StreamAccessKind::SlidingWindow(_), paras) if paras.len() == 1)
        );
    }

    #[test]
    #[should_panic]
    fn test_missing_spawn() {
        let spec = "input a: Int32\n\
        output b (p: Bool) eval with a";
        obtain_expressions(spec);
    }

    #[test]
    #[should_panic]
    fn test_missing_parameters() {
        let spec = "input a: Int32\n\
        output b spawn with a eval with a";
        obtain_expressions(spec);
    }

    #[test]
    #[should_panic]
    fn test_parameter_spawn_mismatch() {
        let spec = "input a: Int32\n\
        output b (p1, p2) spawn with a := a";
        obtain_expressions(spec);
    }

    #[test]
    fn test_expression_context() {
        macro_rules! para_map {
            ($($x:expr),+ $(,)?) => (
                vec![$($x),+].into_iter().map(|(k, set)| (k, set.into_iter().collect::<HashSet<usize>>())).collect::<HashMap<(SRef, usize), HashSet<usize>>>()
            );
        }

        let spec = "input a: Int32\n\
        output b (p1, p2, p3) spawn when a = 5 with (a, a.hold(or: 5), a.offset(by: -5).defaults(to: 7)) eval with a + p1 + p2 + p3\n\
        output c (p1, p2, p3, p4) spawn when a = 5 with (a.hold(or: 5), a, a.hold(or: 5), a.offset(by: -5).defaults(to: 8)) eval with a + p1 + p2 + p3\n\
        output d (p1, p2) spawn with (a.hold(or: 5), a) eval with a + p1 + p2\n";
        let ir = obtain_expressions(spec);
        let ctx = ExpressionContext::new(&ir);

        let b = SRef::Out(0);
        let c = SRef::Out(1);
        let d = SRef::Out(2);

        let b_map = ctx.map_for(b).clone();
        assert_eq!(
            b_map,
            para_map![
                ((b, 0), vec![0]),
                ((b, 1), vec![1]),
                ((b, 2), vec![2]),
                ((c, 0), vec![1]),
                ((c, 1), vec![0]),
                ((c, 2), vec![1]),
            ]
        );

        let c_map = ctx.map_for(c).clone();
        assert_eq!(
            c_map,
            para_map![
                ((c, 0), vec![0, 2]),
                ((c, 1), vec![1]),
                ((c, 2), vec![0, 2]),
                ((c, 3), vec![3]),
                ((b, 0), vec![1]),
                ((b, 1), vec![0, 2]),
            ]
        );

        let d_map = ctx.map_for(d).clone();
        assert_eq!(d_map, para_map![((d, 0), vec![0]), ((d, 1), vec![1])]);
    }
}
