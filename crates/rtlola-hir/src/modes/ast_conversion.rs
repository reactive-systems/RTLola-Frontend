mod naming;

use std::collections::HashMap;
use std::convert::TryInto;
use std::rc::Rc;
use std::time::Duration;

use rtlola_parser::ast;
use rtlola_parser::ast::{FunctionName, Literal as AstLiteral, NodeId, RtLolaAst, SpawnSpec, StreamAccessKind, Type};
use rtlola_reporting::{Handler, Span};

use super::BaseMode;
use crate::hir::{
    AnnotatedPacingType, AnnotatedType, ArithLogOp, Constant as HirConstant, DiscreteAggr, ExprId, Expression,
    ExpressionKind, ExpressionMaps, FnExprKind, Hir, Inlined, Input, InstanceTemplate, Literal, Offset, Output,
    Parameter, SRef, SlidingAggr, SpawnTemplate, StreamAccessKind as IRAccess, Trigger, WRef, WidenExprKind, Window,
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
    pub(crate) fn from_ast(ast: RtLolaAst, handler: &Handler) -> Result<Self, TransformationErr> {
        let mut naming_analyzer = NamingAnalysis::new(&handler);
        let decl_table = naming_analyzer.check(&ast);
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
        ExpressionTransformer::run(decl_table, stream_by_name, ast, func_table)
    }
}

/// The Hir Spawn definition is composed of two optional expressions.
/// The first one refers to the spawn target while the second one represents the spawn condition.
pub type SpawnDef<'a> = (Option<&'a Expression>, Option<&'a Expression>);

/// A [TransformationErr] describes the kind off error raised during the Ast to Hir conversion.
#[derive(Debug, Clone)]
pub enum TransformationErr {
    /// A function was found when a stream was expected.
    InvalidIdentRef(FunctionName),
    /// No valid streamrefernce found while it was expected.
    InvalidRefExpr(String),
    /// A declared constant had no type annotation.
    ConstantWithoutType(Span),
    /// Could not parse numeric literal.
    NonNumericInLiteral(Span),
    /// Invalid activation condition.
    InvalidAc(String),
    /// Offset expression of realtime offset could not be parsed as frequency.
    InvalidRealtimeOffset(Span),
    /// Window duration could not be parsed into correct type.
    InvalidDuration(String, Span),
    /// Missing expression cannot be transformed.
    MissingExpr(Span),
    /// Widen call expects a single type argument.
    MissingWidenArg(Span),
    /// Annotated type could not be matched.
    InvalidTypeArgument(Type, Span),
    /// Called functino unknown.
    UnknownFunction(Span),
    /// Input stream had no type annotation.
    MissingInputType(Span),
    /// Object method called, currently unimplemented.
    MethodAccess,
    /// Non duration literal with postfix found.
    InvalidLiteral(Span),
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
        let RtLolaAst {
            imports: _,   // todo
            constants: _, //handled through naming analysis
            inputs,
            outputs,
            trigger,
            type_declarations: _,
        } = ast;
        let mut exprid_to_expr = HashMap::new();
        let mut hir_outputs = vec![];
        for (ix, o) in outputs.into_iter().enumerate() {
            let sr = SRef::Out(ix);
            let ast::Output {
                expression,
                name,
                params,
                spawn,
                filter,
                close,
                annotated_type,
                annotated_pacing_type,
                ..
            } = (*o).clone();
            let params: Vec<Parameter> = params
                .iter()
                .enumerate()
                .map(|(ix, p)| {
                    assert_eq!(ix, p.param_idx);
                    Parameter {
                        name: p.name.name.clone(),
                        annotated_type: p.ty.as_ref().and_then(Self::annotated_type),
                        idx: p.param_idx,
                        span: p.span.clone(),
                    }
                })
                .collect();
            let annotated_type = annotated_type.as_ref().and_then(Self::annotated_type);
            let expression = self.transform_expression(expression, sr)?;
            let expr_id = expression.eid;
            exprid_to_expr.insert(expr_id, expression);
            let annotated_pacing_type = annotated_pacing_type
                .map_or(Ok(None), |pt| self.transform_pt(&mut exprid_to_expr, pt, sr).map(Some))?;
            let instance_template = self.transform_template_spec(spawn, filter, close, &mut exprid_to_expr, sr)?;
            hir_outputs.push(Output {
                name: name.name,
                sr,
                params,
                instance_template,
                annotated_type,
                annotated_pacing_type,
                expr_id,
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
                        .ok_or_else(|| TransformationErr::MissingInputType(i.span.clone()))?,
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

    fn annotated_type(ast_ty: &Type) -> Option<AnnotatedType> {
        use rtlola_parser::ast::TypeKind;
        match &ast_ty.kind {
            TypeKind::Tuple(vec) => {
                Some(AnnotatedType::Tuple(
                    vec.iter()
                        .map(|inner| Self::annotated_type(inner).expect("Inner types can not be missing"))
                        .collect(),
                ))
            },
            TypeKind::Optional(inner) => {
                Some(AnnotatedType::Option(
                    Self::annotated_type(inner)
                        .expect("Inner types can not be missing")
                        .into(),
                ))
            },
            TypeKind::Simple(string) => {
                if string == "String" {
                    return Some(AnnotatedType::String);
                }
                if string == "Bool" {
                    return Some(AnnotatedType::Bool);
                }
                if let Some(size_str) = string.strip_prefix("Int") {
                    if string.len() == 3 {
                        return Some(AnnotatedType::Int(8));
                    } else {
                        let size: u32 = size_str.parse().expect("Invalid char followed Int type annotation");
                        return Some(AnnotatedType::Int(size));
                    }
                }
                if let Some(size_str) = string.strip_prefix("UInt") {
                    if string.len() == 4 {
                        return Some(AnnotatedType::Int(8));
                    } else {
                        let size: u32 = size_str.parse().expect("Invalid char followed UInt type annotation");
                        return Some(AnnotatedType::UInt(size));
                    }
                }
                if let Some(size_str) = string.strip_prefix("Float") {
                    if string.len() == 5 {
                        return Some(AnnotatedType::Int(8));
                    } else {
                        let size: u32 = size_str.parse().expect("Invalid char followed Float type annotation");
                        return Some(AnnotatedType::Float(size));
                    }
                }
                if string == "Bytes" {
                    return Some(AnnotatedType::Bytes);
                }
                None
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
                    _ => {
                        Err(TransformationErr::InvalidRefExpr(String::from(
                            "Non-identifier transformed to SRef",
                        )))
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
                    _ => Err(TransformationErr::InvalidIdentRef(name.clone())),
                }
            },
            _ => Err(TransformationErr::InvalidRefExpr(format!("{:?}", expr.kind))),
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
                let val = pt_expr.parse_freqspec().map_err(TransformationErr::InvalidAc)?;
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
                            .ok_or_else(|| TransformationErr::ConstantWithoutType(span.clone()))?;
                        ExpressionKind::LoadConstant(HirConstant::Inlined(Inlined {
                            lit: self.transform_literal(&c.literal)?,
                            ty: annotated_type,
                        }))
                    },

                    Declaration::Param(p) => ExpressionKind::ParameterAccess(current_output, p.param_idx),
                    Declaration::ParamOut(_) | Declaration::Func(_) | Declaration::Type(_) => {
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
                    ast::Offset::Discrete(i) if i > 0 => Some(Offset::FutureDiscrete(i.abs() as u32)),
                    ast::Offset::Discrete(i) => Some(Offset::PastDiscrete(i.abs() as u32)),
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
                let (sref, _) = self.get_stream_ref(&w_expr, current_output)?;
                let idx = self.sliding_windows.len();
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
                ExpressionKind::StreamAccess(sref, IRAccess::DiscreteWindow(WRef::Discrete(idx)), Vec::new())
            },
            ast::ExpressionKind::SlidingWindowAggregation {
                expr: w_expr,
                duration,
                wait,
                aggregation: win_op,
            } => {
                let (sref, _) = self.get_stream_ref(&w_expr, current_output)?;
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
                ExpressionKind::StreamAccess(sref, IRAccess::SlidingWindow(WRef::Sliding(idx)), Vec::new())
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
                                    .ok_or_else(|| TransformationErr::InvalidTypeArgument(t.clone(), span.clone()))?
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
                                    .ok_or_else(|| TransformationErr::InvalidTypeArgument(t, span.clone()))
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

    fn transform_template_spec(
        &mut self,
        spawn_spec: Option<SpawnSpec>,
        filter_spec: Option<ast::FilterSpec>,
        close_spec: Option<ast::CloseSpec>,
        exprid_to_expr: &mut HashMap<ExprId, Expression>,
        current_output: SRef,
    ) -> Result<InstanceTemplate, TransformationErr> {
        Ok(InstanceTemplate {
            spawn: spawn_spec.map_or(Ok(None), |spawn_spec| {
                let SpawnSpec {
                    target,
                    annotated_pacing,
                    condition,
                    is_if,
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
                    let mut e = self.transform_expression(cond_expr, current_output)?;
                    if !is_if {
                        e = Expression {
                            kind: ExpressionKind::ArithLog(ArithLogOp::Not, vec![e.clone()]),
                            eid: self.next_exp_id(),
                            span: e.span,
                        }
                    }
                    Ok(Some(Self::insert_return(exprid_to_expr, e)))
                })?;
                Ok(Some(SpawnTemplate {
                    target,
                    pacing,
                    condition,
                }))
            })?,
            filter: filter_spec.map_or(Ok(None), |filter_spec| {
                Ok(Some(Self::insert_return(
                    exprid_to_expr,
                    self.transform_expression(filter_spec.target, current_output)?,
                )))
            })?,
            close: close_spec.map_or(Ok(None), |close_spec| {
                Ok(Some(Self::insert_return(
                    exprid_to_expr,
                    self.transform_expression(close_spec.target, current_output)?,
                )))
            })?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rtlola_parser::ast::WindowOperation;
    use rtlola_parser::{parse_with_handler, ParserConfig};

    use super::*;
    use crate::hir::StreamAccessKind;

    fn obtain_expressions(spec: &str) -> Hir<BaseMode> {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let ast = parse_with_handler(ParserConfig::for_string(spec.to_string()), &handler)
            .unwrap_or_else(|e| panic!("{}", e));
        crate::from_ast(ast, &handler).unwrap()
    }

    #[test]
    fn all() {
        //Tests all cases are implemented
        let spec = "
        import math
        input i: Int8
        output o := 3
        output o2 @1Hz := 4
        output o3 := if true then 1 else 2
        output o4 := 3 + 4
        output o5 := o
        output f := sqrt(4)
        output p (x,y) := x
        output t := (1,3)
        output t1 := t.0
        output off := o.defaults(to:2)
        output off := o.offset(by:-1)
        output w := o2.aggregate(over:3s , using: sum)";
        let _ir = obtain_expressions(spec);
        //TODO
    }

    #[test]
    fn transform_default() {
        let spec = "input o :Int8 output off := o.defaults(to:-1)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].expr_id;
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
            let output_expr_id = ir.outputs[0].expr_id;
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
        let output_expr_id = ir.outputs[0].expr_id;
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
        let spec = "output o(a,b,c) :=  if c then a else b";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].expr_id;
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
        let spec = "output o(a,b,c) :=  if c then a else b output A := o(1,2,true).offset(by:-1)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[1].expr_id;
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
        let output_expr_id = ir.outputs[0].expr_id;
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
        let output_expr_id = ir.outputs[0].expr_id;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(expr.kind, ExpressionKind::TupleAccess(_, 1)));
    }

    #[test]
    fn simple_trigger() {
        let spec = "trigger true";
        let ir = obtain_expressions(spec);
        assert_eq!(ir.num_triggers(), 1);
        let tr = &ir.triggers[0];
        let expr = &ir.expr(tr.sr);
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
        let expr = ir.expr(tr.sr);
        assert!(matches!(expr.kind, ExpressionKind::ArithLog(ArithLogOp::Eq, _)));
    }

    #[test]
    fn arith_op() {
        use crate::hir::ArithLogOp;
        let spec = "output o := 3 + 5 ";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].expr_id;
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
        let spec = "import math output o(a: Int) := max(3,4) output c := o(1)";
        let ir = obtain_expressions(spec);
        //check that this functions exists
        let _ = ir.func_declaration("max");
        let output_expr_id = ir.outputs[1].expr_id;
        let expr = &ir.expression(output_expr_id);
        assert!(matches!(
            expr.kind,
            ExpressionKind::StreamAccess(SRef::Out(0), StreamAccessKind::Sync, _)
        ));
    }

    #[test]
    fn function_param_default() {
        let spec = "import math output o(a: Int) := sqrt(a) output c := o(1).defaults(to:1)";
        let ir = obtain_expressions(spec);
        //check purely for valid access
        let _ = ir.func_declaration("sqrt");
        let output_expr_id = ir.outputs[1].expr_id;
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
        let spec = "input i: Bool output c(a: Int8): Bool @1Hz spawn with i if i filter i close i:= i";
        let ir = obtain_expressions(spec);
        let output: Output = ir.outputs[0].clone();
        assert!(output.annotated_type.is_some());
        assert!(matches!(
            output.annotated_pacing_type,
            Some(AnnotatedPacingType::Frequency { .. })
        ));
        assert!(output.params.len() == 1);
        let fil: &Expression = ir.filter(output.sr).unwrap();
        assert!(matches!(
            fil.kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
        let close = ir.close(output.sr).unwrap();
        assert!(matches!(
            close.kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
        let (invoke, op_cond) = ir.spawn(output.sr).unwrap();
        assert!(matches!(
            invoke.unwrap().kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
        assert!(matches!(
            op_cond.unwrap().kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)
        ));
    }

    #[test]
    fn test_ac() {
        use crate::hir::{AnnotatedPacingType, Output};
        let spec = "input in: Bool output a: Bool @2.5Hz := true output b: Bool @in := false";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(matches!(
            a.annotated_pacing_type,
            Some(AnnotatedPacingType::Frequency { .. })
        ));
        let b: &Output = &ir.outputs[1];
        assert!(matches!(b.annotated_pacing_type, Some(AnnotatedPacingType::Expr(_))));
    }

    #[test]
    fn test_spawn_missing_exp() {
        use crate::hir::{Output, SpawnTemplate};
        let spec = "output a: Bool @2.5Hz spawn with () if true := true";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(matches!(
            a.instance_template.spawn,
            Some(SpawnTemplate { target: None, .. })
        ));
    }

    #[test]
    fn test_spawn_pacing() {
        use crate::hir::Output;
        let spec = "output a: Bool @2.5Hz spawn @1Hz with () if true := true";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(a.instance_template.spawn.is_some());
        let template = a.instance_template.spawn;
        assert!(matches!(
            template.unwrap().pacing,
            Some(AnnotatedPacingType::Frequency { .. })
        ));
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
}
