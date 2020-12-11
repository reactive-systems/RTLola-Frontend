use crate::{
    common_ir::StreamAccessKind as IRAccess,
    hir::expression::{
        Constant as HIRConstant, ConstantLiteral, DiscreteWindow, ExprId, Expression, ExpressionKind, SlidingWindow,
    },
    hir::modes::{HirMode, IrExprRes},
    hir::AC,
    hir::{AnnotatedType, Hir, Input, InstanceTemplate, Output, Parameter, SpawnTemplate, Trigger, Window},
};

use super::IrExpression;
use crate::analysis::naming::{Declaration, NamingAnalysis};
use crate::ast;
use crate::ast::{Ast, Literal, StreamAccessKind, Type};
use crate::common_ir::{Offset, SRef, WRef};
use crate::hir::function_lookup::FuncDecl;
use crate::parse::NodeId;
use crate::reporting::Handler;
use crate::FrontendConfig;
use itertools::{Either, Itertools};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;

pub trait WithIrExpr {
    fn window_refs(&self) -> Vec<Window>;
    fn all_windows(&self) -> (Vec<SlidingWindow>, Vec<DiscreteWindow>) {
        self.window_refs().into_iter().partition_map(|w| self.single_window(w))
    }
    fn single_window(&self, window: Window) -> Either<SlidingWindow, DiscreteWindow>;
    fn expression(&self, id: ExprId) -> &Expression;
    fn func_declaration(&self, func_name: &str) -> &FuncDecl;
}
impl WithIrExpr for IrExprRes {
    fn window_refs(&self) -> Vec<Window> {
        self.windows.values().map(|w| Window { expr: w.eid }).collect()
    }

    fn single_window(&self, window: Window) -> Either<SlidingWindow, DiscreteWindow> {
        Either::Left(self.windows[&window.expr])
    }

    fn expression(&self, id: ExprId) -> &Expression {
        &self.exprid_to_expr[&id]
    }

    fn func_declaration(&self, func_name: &str) -> &FuncDecl {
        &self.func_table[func_name]
    }
}

pub(crate) type SpawnDef<'a> = (Option<&'a Expression>, Option<&'a Expression>);

impl<M> Hir<M>
where
    M: WithIrExpr + HirMode + 'static,
{
    pub fn windows(&self) -> Vec<Window> {
        self.window_refs()
    }

    pub fn expr(&self, sr: SRef) -> &Expression {
        match sr {
            SRef::InRef(_) => unimplemented!("No Expression access for input streams possible"),
            SRef::OutRef(o) => {
                if o < self.outputs.len() {
                    let output = self.outputs.iter().find(|o| o.sr == sr);
                    let id = output.expect("Accessing non-existing Output-Stream").expr_id;
                    self.mode.expression(id)
                } else {
                    let tr = self.triggers.iter().find(|tr| tr.sr == sr);
                    let id = tr.expect("Accessing non-existing Trigger").expr_id;
                    self.mode.expression(id)
                }
            }
        }
    }

    pub fn act_cond(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::InRef(_) => None,
            SRef::OutRef(o) => {
                if o < self.outputs.len() {
                    let output = self.outputs.iter().find(|o| o.sr == sr);
                    if let Some(ac) = output.and_then(|o| o.activation_condition.as_ref()) {
                        match ac {
                            AC::Expr(e) => Some(self.mode.expression(*e)),
                            AC::Frequency { .. } => None, //May change return type
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    pub fn spawn(&self, sr: SRef) -> Option<SpawnDef> {
        match sr {
            SRef::InRef(_) => None,
            SRef::OutRef(o) => {
                if o < self.outputs.len() {
                    let output = self.outputs.iter().find(|o| o.sr == sr);
                    output.and_then(|o| {
                        o.instance_template.spawn.as_ref().map(|st| {
                            (st.target.map(|e| self.mode.expression(e)), st.condition.map(|e| self.mode.expression(e)))
                        })
                    })
                } else {
                    None
                }
            }
        }
    }
    pub fn filter(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::InRef(_) => None,
            SRef::OutRef(o) => {
                if o < self.outputs.len() {
                    let output = self.outputs.iter().find(|o| o.sr == sr);
                    output.and_then(|o| o.instance_template.filter.map(|e| self.mode.expression(e)))
                } else {
                    None
                }
            }
        }
    }
    pub fn close(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::InRef(_) => None,
            SRef::OutRef(o) => {
                if o < self.outputs.len() {
                    let output = self.outputs.iter().find(|o| o.sr == sr);
                    output.and_then(|o| o.instance_template.close.map(|e| self.mode.expression(e)))
                } else {
                    None
                }
            }
        }
    }
}

pub trait IrExprWrapper {
    type InnerE: WithIrExpr;
    fn inner_expr(&self) -> &Self::InnerE;
}

impl IrExprWrapper for IrExpression {
    type InnerE = IrExprRes;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr_res
    }
}

impl<A: IrExprWrapper<InnerE = T>, T: WithIrExpr + 'static> WithIrExpr for A {
    fn window_refs(&self) -> Vec<Window> {
        self.inner_expr().window_refs()
    }
    fn single_window(&self, window: Window) -> Either<SlidingWindow, DiscreteWindow> {
        self.inner_expr().single_window(window)
    }

    fn expression(&self, id: ExprId) -> &Expression {
        self.inner_expr().expression(id)
    }
    fn func_declaration(&self, func_name: &str) -> &FuncDecl {
        self.inner_expr().func_declaration(func_name)
    }
}

#[derive(Debug)]
pub enum TransformationError {
    InvalidIdentRef(Declaration),
    InvalidRefExpr(String),
}

#[derive(Debug)]
pub struct ExpressionTransformer {
    sliding_windows: Vec<SlidingWindow>,
    discrete_windows: Vec<DiscreteWindow>,
    windows: Vec<Window>,
    decl_table: HashMap<NodeId, Declaration>,
    stream_by_name: HashMap<String, SRef>,
    current_exp_id: u32,
}

impl ExpressionTransformer {
    fn new(decl_table: HashMap<NodeId, Declaration>, stream_by_name: HashMap<String, SRef>) -> Self {
        ExpressionTransformer {
            sliding_windows: vec![],
            discrete_windows: vec![],
            windows: vec![],
            decl_table,
            stream_by_name,
            current_exp_id: 0,
        }
    }

    fn get_stream_ref(
        &mut self,
        expr: &ast::Expression,
        current_output: SRef,
    ) -> Result<(SRef, Vec<Expression>), TransformationError> {
        match &expr.kind {
            ast::ExpressionKind::Ident(_) => match &self.decl_table[&expr.id] {
                Declaration::In(i) => Ok((self.stream_by_name[&i.name.name], Vec::new())),
                Declaration::Out(o) => Ok((self.stream_by_name[&o.name.name], Vec::new())),
                _ => Err(TransformationError::InvalidRefExpr(String::from("Non-identifier transformed to SRef"))),
            },
            ast::ExpressionKind::Function(_, _, args) => match &self.decl_table[&expr.id] {
                Declaration::ParamOut(o) => Ok((
                    self.stream_by_name[&o.name.name],
                    args.iter().map(|e| self.transform_expression(*e.clone(), current_output)).collect(),
                )),
                decl => Err(TransformationError::InvalidIdentRef(decl.clone())),
            },
            _ => Err(TransformationError::InvalidRefExpr(format!("{:?}", expr.kind))),
        }
    }

    fn next_exp_id(&mut self) -> ExprId {
        let ret = self.current_exp_id;
        self.current_exp_id += 1;
        ExprId(ret)
    }

    fn transform_literal(&self, lit: &Literal) -> ConstantLiteral {
        match &lit.kind {
            ast::LitKind::Bool(b) => ConstantLiteral::Bool(*b),
            ast::LitKind::Str(s) | ast::LitKind::RawStr(s) => ConstantLiteral::Str(s.clone()),
            ast::LitKind::Numeric(num_str, postfix) => {
                match postfix {
                    None => {
                        if num_str.contains('.') {
                            // Floating Point
                            ConstantLiteral::Float(num_str.parse().expect("Expect numeric value for literal"))
                        } else if num_str.starts_with('-') {
                            ConstantLiteral::SInt(num_str.parse().expect("Expect numeric value for literal"))
                        } else {
                            ConstantLiteral::Integer(num_str.parse().expect("Expect numeric value for literal"))
                        }
                    }
                    Some(s) if s.is_empty() => {
                        if num_str.contains('.') {
                            // Floating Point
                            ConstantLiteral::Float(num_str.parse().expect("Expect numeric value for literal"))
                        } else if num_str.starts_with('-') {
                            ConstantLiteral::SInt(num_str.parse().expect("Expect numeric value for literal"))
                        } else {
                            ConstantLiteral::Integer(num_str.parse().expect("Expect numeric value for literal"))
                        }
                    }
                    Some(_) => unreachable!(),
                }
            }
        }
    }

    fn transform_ac(
        &mut self,
        exprid_to_expr: &mut HashMap<ExprId, Expression>,
        ac_expr: ast::Expression,
        current: SRef,
    ) -> AC {
        if let ast::ExpressionKind::Lit(l) = &ac_expr.kind {
            if let ast::LitKind::Numeric(_, Some(_)) = &l.kind {
                return AC::Frequency {
                    span: ac_expr.span.clone(),
                    value: ac_expr.parse_freqspec().expect("Invalid Literal with Postfix (unable to parse Frequency)"),
                };
            }
        }
        AC::Expr(insert_return(exprid_to_expr, self.transform_expression(ac_expr, current)))
    }

    fn transform_expression(&mut self, ast_expression: ast::Expression, current_output: SRef) -> Expression {
        let new_id = self.next_exp_id();
        let span = ast_expression.span;
        let kind: ExpressionKind = match ast_expression.kind {
            ast::ExpressionKind::Lit(lit) => {
                let constant = self.transform_literal(&lit);
                ExpressionKind::LoadConstant(HIRConstant::BasicConstant(constant))
            }
            ast::ExpressionKind::Ident(_) => match &self.decl_table[&ast_expression.id] {
                Declaration::Out(o) => {
                    let sr = self.stream_by_name[&o.name.name];
                    ExpressionKind::StreamAccess(sr, IRAccess::Sync, Vec::new())
                }
                Declaration::In(i) => {
                    let sr = self.stream_by_name[&i.name.name];
                    ExpressionKind::StreamAccess(sr, IRAccess::Sync, Vec::new())
                }
                Declaration::Const(c) => {
                    let annotated_type =
                        match annotated_type(&c.ty.as_ref().expect("Constant variables must have type annotation")) {
                            Some(t) => t,
                            None => unreachable!("Constant variables must have type annotation"),
                        };
                    ExpressionKind::LoadConstant(HIRConstant::InlinedConstant(
                        self.transform_literal(&c.literal),
                        annotated_type,
                    ))
                }

                Declaration::Param(p) => ExpressionKind::ParameterAccess(current_output, p.param_idx),
                Declaration::ParamOut(_) => todo!("may be unreachable"),
                Declaration::Func(_) => todo!("unreach"),
                Declaration::Type(_) => todo!("type identifier transform"),
            },
            ast::ExpressionKind::StreamAccess(expr, kind) => {
                let access_kind = if let StreamAccessKind::Hold = kind { IRAccess::Hold } else { IRAccess::Sync }; //TODO
                let (expr_ref, args) =
                    self.get_stream_ref(&*expr, current_output).expect("Invalid streamaccess expression"); //TODO error case
                                                                                                           //TODO param function case
                ExpressionKind::StreamAccess(expr_ref, access_kind, args)
            }
            ast::ExpressionKind::Default(expr, def) => ExpressionKind::Default {
                expr: self.transform_expression(*expr, current_output).into(),
                default: self.transform_expression(*def, current_output).into(),
            },
            ast::ExpressionKind::Offset(ref target_expr, offset) => {
                use uom::si::time::nanosecond;
                let ir_offset = match offset {
                    ast::Offset::Discrete(i) if i == 0 => None,
                    ast::Offset::Discrete(i) if i > 0 => Some(Offset::FutureDiscreteOffset(i.abs() as u32)),
                    ast::Offset::Discrete(i) => Some(Offset::PastDiscreteOffset(i.abs() as u32)),
                    ast::Offset::RealTime(_, _) => {
                        let offset_uom_time =
                            offset.to_uom_time().expect("ast::Offset::RealTime should return uom_time");
                        let dur = offset_uom_time.get::<nanosecond>().to_integer();
                        //TODO FIXME check potential loss of precision
                        let time = offset_uom_time.get::<nanosecond>();
                        let numer = time.numer();
                        match numer {
                            0 => None,
                            i if i < &0 => {
                                let positive_dur = Duration::from_nanos((-dur) as u64);
                                Some(Offset::PastRealTimeOffset(positive_dur))
                            }
                            _ => {
                                let positive_dur = Duration::from_nanos(dur as u64);
                                Some(Offset::FutureRealTimeOffset(positive_dur))
                            }
                        }
                    }
                };
                let (expr_ref, args) = self.get_stream_ref(&*target_expr, current_output).unwrap(); //TODO error case
                let kind = ir_offset.map(IRAccess::Offset).unwrap_or(IRAccess::Sync);
                ExpressionKind::StreamAccess(expr_ref, kind, args)
            }
            ast::ExpressionKind::DiscreteWindowAggregation { expr: w_expr, duration, wait, aggregation: win_op } => {
                if let Ok((sref, _)) = self.get_stream_ref(&w_expr, current_output) {
                    let idx = self.sliding_windows.len();
                    let wref = WRef::SlidingRef(idx);
                    let duration = (*duration).parse_discrete_duration().expect("Todo Error case");
                    self.windows.push(Window { expr: new_id });
                    let window = DiscreteWindow {
                        target: sref,
                        caller: current_output,
                        duration: duration as u32,
                        wait,
                        op: win_op,
                        reference: wref,
                        eid: new_id,
                    };
                    self.discrete_windows.push(window);
                    ExpressionKind::StreamAccess(sref, IRAccess::DiscreteWindow(WRef::DiscreteRef(idx)), Vec::new())
                } else {
                    todo!("error case")
                }
            }
            ast::ExpressionKind::SlidingWindowAggregation { expr: w_expr, duration, wait, aggregation: win_op } => {
                if let Ok((sref, _)) = self.get_stream_ref(&w_expr, current_output) {
                    let idx = self.sliding_windows.len();
                    let wref = WRef::SlidingRef(idx);
                    let duration = parse_duration_from_expr(&*duration);
                    self.windows.push(Window { expr: new_id });
                    let window = SlidingWindow {
                        target: sref,
                        caller: current_output,
                        duration,
                        wait,
                        op: win_op,
                        reference: wref,
                        eid: new_id,
                    };
                    self.sliding_windows.push(window);
                    ExpressionKind::StreamAccess(sref, IRAccess::SlidingWindow(WRef::SlidingRef(idx)), Vec::new())
                } else {
                    todo!("error case")
                }
            }
            ast::ExpressionKind::Binary(op, left, right) => {
                use crate::ast::BinOp;
                use crate::hir::expression::ArithLogOp::*;
                let arith_op = match op {
                    BinOp::Add => Add,
                    BinOp::Sub => Sub,
                    BinOp::Mul => Mul,
                    BinOp::Div => Div,
                    BinOp::Rem => Rem,
                    BinOp::Pow => Pow,
                    BinOp::And => Add,
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
                    self.transform_expression(*left, current_output),
                    self.transform_expression(*right, current_output),
                ];
                ExpressionKind::ArithLog(arith_op, arguments)
            }
            ast::ExpressionKind::Unary(op, arg) => {
                use crate::ast::UnOp;
                use crate::hir::expression::ArithLogOp::*;
                let arith_op = match op {
                    UnOp::Not => Not,
                    UnOp::Neg => Neg,
                    UnOp::BitNot => BitNot,
                };
                let arguments: Vec<Expression> = vec![self.transform_expression(*arg, current_output)];
                ExpressionKind::ArithLog(arith_op, arguments)
            }
            ast::ExpressionKind::Ite(cond, cons, alt) => {
                let condition = self.transform_expression(*cond, current_output).into();
                let consequence = self.transform_expression(*cons, current_output).into();
                let alternative = self.transform_expression(*alt, current_output).into();
                ExpressionKind::Ite { condition, consequence, alternative }
            }
            ast::ExpressionKind::ParenthesizedExpression(_, inner, _) => {
                return self.transform_expression(*inner, current_output);
            }
            ast::ExpressionKind::MissingExpression => unimplemented!(),
            ast::ExpressionKind::Tuple(inner) => ExpressionKind::Tuple(
                inner.into_iter().map(|ex| self.transform_expression(*ex, current_output)).collect(),
            ),
            ast::ExpressionKind::Field(inner_exp, ident) => {
                let num: usize = ident.name.parse().expect("checked in AST verifier");
                let inner = self.transform_expression(*inner_exp, current_output).into();
                ExpressionKind::TupleAccess(inner, num)
            }
            ast::ExpressionKind::Method(_base, _name, _types, _params) => todo!(),
            ast::ExpressionKind::Function(name, type_param, args) => {
                let decl = self.decl_table.get(&ast_expression.id);
                if let Some(decl) = decl {
                    match decl {
                        Declaration::Func(_) => {
                            let name = name.name.name;
                            let args: Vec<Expression> =
                                args.into_iter().map(|ex| self.transform_expression(*ex, current_output)).collect();

                            if name.starts_with("widen") {
                                ExpressionKind::Widen(
                                    (args.get(0).expect("Widen is expecting exactly 1 Argument").clone()).into(),
                                    match type_param.get(0) {
                                        Some(t) => {
                                            annotated_type(t).expect("given type arguments have to be replaceable")
                                        }
                                        None => todo!("error case"),
                                    },
                                )
                            } else {
                                ExpressionKind::Function {
                                    name,
                                    args,
                                    type_param: type_param
                                        .into_iter()
                                        .map(|t| {
                                            annotated_type(&t).expect("given type arguments have to be replaceable")
                                        })
                                        .collect(),
                                }
                            }
                        }
                        Declaration::ParamOut(_) => ExpressionKind::StreamAccess(
                            self.stream_by_name[&name.name.name],
                            IRAccess::Sync,
                            args.into_iter().map(|ex| self.transform_expression(*ex, current_output)).collect(),
                        ),
                        _ => todo!("error case"),
                    }
                } else {
                    todo!("Error case: unknown function")
                }
            }
        };
        Expression { kind, eid: new_id, span }
    }
}

fn parse_duration_from_expr(ast_expression: &ast::Expression) -> Duration {
    use num::{traits::Inv, ToPrimitive};
    use uom::si::frequency::hertz;
    use uom::si::rational64::Time as UOM_Time;
    use uom::si::time::second;

    let freq = ast_expression.parse_freqspec().unwrap();
    let period = UOM_Time::new::<second>(freq.get::<hertz>().inv());
    Duration::from_nanos(period.get::<uom::si::time::nanosecond>().to_integer().to_u64().unwrap())
}

fn insert_return(exprid_to_expr: &mut HashMap<ExprId, Expression>, expr: Expression) -> ExprId {
    let id = expr.eid;
    exprid_to_expr.insert(id, expr);
    id
}

fn transform_template_spec(
    transformer: &mut ExpressionTransformer,
    spawn_spec: Option<ast::SpawnSpec>,
    filter_spec: Option<ast::FilterSpec>,
    close_spec: Option<ast::CloseSpec>,
    exprid_to_expr: &mut HashMap<ExprId, Expression>,
    current_output: SRef,
) -> InstanceTemplate {
    InstanceTemplate {
        spawn: spawn_spec.map(|spawn_spec| SpawnTemplate {
            target: spawn_spec.target.and_then(|target| {
                if let ast::ExpressionKind::ParenthesizedExpression(_, ref exp, _) = target.kind {
                    if let ast::ExpressionKind::MissingExpression = exp.kind {
                        return None;
                    }
                }
                Some(insert_return(exprid_to_expr, transformer.transform_expression(target, current_output)))
            }),
            condition: spawn_spec.condition.map(|cond_expr| {
                insert_return(exprid_to_expr, transformer.transform_expression(cond_expr, current_output))
            }),
            is_if: spawn_spec.is_if,
        }),
        filter: filter_spec.map(|filter_spec| {
            insert_return(exprid_to_expr, transformer.transform_expression(filter_spec.target, current_output))
        }),
        close: close_spec.map(|close_spec| {
            insert_return(exprid_to_expr, transformer.transform_expression(close_spec.target, current_output))
        }),
    }
}

impl Hir<IrExpression> {
    pub fn transform_expressions(ast: Ast, handler: &Handler, config: &FrontendConfig) -> Self {
        let mut naming_analyzer = NamingAnalysis::new(&handler, *config);
        let decl_table = naming_analyzer.check(&ast);
        let func_table: HashMap<String, FuncDecl> = decl_table
            .values()
            .filter(|decl| matches!(decl, Declaration::Func(_)))
            .map(|decl| {
                if let Declaration::Func(fun_decl) = decl {
                    (fun_decl.name.name.name.clone(), (**fun_decl).clone())
                } else {
                    unreachable!("assured by filter")
                }
            })
            .collect();
        let Ast {
            imports: _,   // todo
            constants: _, //handled through naming analysis
            inputs,
            outputs,
            trigger,
            type_declarations: _,
        } = ast;

        let mut hir_outputs = vec![];
        let mut stream_by_name = HashMap::new();
        let mut exprid_to_expr = HashMap::new();

        for (ix, o) in outputs.iter().enumerate() {
            let sr = SRef::OutRef(ix);
            stream_by_name.insert(o.name.name.clone(), sr);
        }
        for (ix, i) in inputs.iter().enumerate() {
            let sr = SRef::InRef(ix);
            stream_by_name.insert(i.name.name.clone(), sr);
        }
        let stream_by_name = stream_by_name;
        let mut expr_transformer = ExpressionTransformer::new(decl_table, stream_by_name);

        for (ix, o) in outputs.into_iter().enumerate() {
            let sr = SRef::OutRef(ix);
            let ast::Output { expression, name, params, spawn, filter, close, ty, extend, .. } = (*o).clone();
            let params: Vec<Parameter> = params
                .iter()
                .enumerate()
                .map(|(ix, p)| {
                    assert_eq!(ix, p.param_idx);
                    Parameter {
                        name: p.name.name.clone(),
                        annotated_type: annotated_type(&p.ty),
                        idx: p.param_idx,
                        span: p.span.clone(),
                    }
                })
                .collect();
            let annotated_type = annotated_type(&ty);
            let expression = expr_transformer.transform_expression(expression, sr);
            let expr_id = expression.eid;
            exprid_to_expr.insert(expr_id, expression);
            let ac = extend.expr.map(|ex| expr_transformer.transform_ac(&mut exprid_to_expr, ex, sr));
            let instance_template =
                transform_template_spec(&mut expr_transformer, spawn, filter, close, &mut exprid_to_expr, sr);
            hir_outputs.push(Output {
                name: name.name,
                sr,
                params,
                instance_template,
                annotated_type,
                activation_condition: ac,
                expr_id,
                span: o.span.clone(),
            });
        }
        let hir_outputs = hir_outputs;
        let mut hir_triggers = vec![];
        for (ix, t) in trigger.into_iter().enumerate() {
            let sr = SRef::OutRef(hir_outputs.len() + ix);
            let ast::Trigger { message, name, expression, span, .. } =
                Rc::try_unwrap(t).expect("other strong references should be dropped now");
            let expr_id = insert_return(&mut exprid_to_expr, expr_transformer.transform_expression(expression, sr));

            hir_triggers.push(Trigger::new(name, message, expr_id, sr, span));
        }
        let hir_triggers = hir_triggers;
        let hir_inputs: Vec<Input> = inputs
            .into_iter()
            .enumerate()
            .map(|(ix, i)| Input {
                annotated_type: annotated_type(&i.ty).expect("Input Streams must have type annotation"),
                name: i.name.name.clone(),
                sr: SRef::InRef(ix),
                span: i.span.clone(),
            })
            .collect();

        let ExpressionTransformer { sliding_windows, discrete_windows: _, .. } = expr_transformer;
        //TODO use discrete windows
        let windows: HashMap<ExprId, SlidingWindow> = sliding_windows.into_iter().map(|w| (w.eid, w)).collect();

        let new_mode = IrExpression { ir_expr_res: IrExprRes { exprid_to_expr, windows, func_table } };

        Hir {
            next_input_ref: hir_inputs.len(),
            inputs: hir_inputs,
            next_output_ref: hir_outputs.len(),
            outputs: hir_outputs,
            triggers: hir_triggers,
            mode: new_mode,
        }
    }
}

pub fn annotated_type(ast_ty: &Type) -> Option<AnnotatedType> {
    use crate::ast::TypeKind;
    match &ast_ty.kind {
        TypeKind::Tuple(vec) => Some(AnnotatedType::Tuple(
            vec.iter().map(|inner| annotated_type(inner).expect("Inner types can not be missing")).collect(),
        )),
        TypeKind::Optional(inner) => {
            Some(AnnotatedType::Option(annotated_type(inner).expect("Inner types can not be missing").into()))
        }
        TypeKind::Simple(string) => {
            if string == "String" {
                return Some(AnnotatedType::String);
            }
            if string == "Bool" {
                return Some(AnnotatedType::Bool);
            }
            if string.starts_with("Int") {
                if string.len() == 3 {
                    return Some(AnnotatedType::Int(8));
                } else {
                    let size: u32 = string[3..].parse().expect("Invalid char followed Int type annotation");
                    return Some(AnnotatedType::Int(size));
                }
            }
            if string.starts_with("UInt") {
                if string.len() == 4 {
                    return Some(AnnotatedType::Int(8));
                } else {
                    let size: u32 = string[4..].parse().expect("Invalid char followed UInt type annotation");
                    return Some(AnnotatedType::UInt(size));
                }
            }
            if string.starts_with("Float") {
                if string.len() == 5 {
                    return Some(AnnotatedType::Int(8));
                } else {
                    let size: u32 = string[5..].parse().expect("Invalid char followed Float type annotation");
                    return Some(AnnotatedType::Float(size));
                }
            }
            if string == "Bytes" {
                return Some(AnnotatedType::Bytes);
            }
            None
        }
        TypeKind::Inferred => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::WindowOperation;
    use crate::common_ir::StreamAccessKind;
    use crate::parse::parse;
    use std::path::PathBuf;

    fn obtain_expressions(spec: &str) -> Hir<IrExpression> {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        let replaced: Hir<IrExpression> = Hir::<IrExpression>::transform_expressions(ast, &handler, &config);
        replaced
    }

    #[test]
    fn window_len() {
        let ir = obtain_expressions("output a @1Hz := 1 output b @1min:= a.aggregate(over: 1s, using: sum)");
        assert_eq!(1, ir.mode.ir_expr_res.windows.len());
        //TODO
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
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::Default{..}));
    }

    #[test]
    fn transform_offset() {
        use crate::common_ir::StreamAccessKind;
        //TODO do remaining cases
        for (spec, offset) in &[
            ("input o :Int8 output off := o", StreamAccessKind::Sync),
            //("input o :Int8 output off := o.aggregate(over: 1s, using: sum)",StreamAccessKind::DiscreteWindow(WRef::SlidingRef(0))),
            ("input o :Int8 output off := o.hold()", StreamAccessKind::Hold),
            ("input o :Int8 output off := o.offset(by:-1)", StreamAccessKind::Offset(Offset::PastDiscreteOffset(1))),
            ("input o :Int8 output off := o.offset(by: 1)", StreamAccessKind::Offset(Offset::FutureDiscreteOffset(1))),
            ("input o :Int8 output off := o.offset(by: 0)", StreamAccessKind::Sync),
            (
                "input o :Int8 output off := o.offset(by:-1s)",
                StreamAccessKind::Offset(Offset::PastRealTimeOffset(Duration::from_secs(1))),
            ),
            (
                "input o :Int8 output off := o.offset(by: 1s)",
                StreamAccessKind::Offset(Offset::FutureRealTimeOffset(Duration::from_secs(1))),
            ),
            ("input o :Int8 output off := o.offset(by: 0s)", StreamAccessKind::Sync),
        ] {
            let ir = obtain_expressions(spec);
            let output_expr_id = ir.outputs[0].expr_id;
            let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
            assert!(matches!(expr.kind, ExpressionKind::StreamAccess(SRef::InRef(0), _, _)));
            if let ExpressionKind::StreamAccess(SRef::InRef(0), result_kind, _) = expr.kind {
                assert_eq!(result_kind, *offset);
            }
        }
    }

    #[test]
    fn transform_aggr() {
        let spec = "input i:Int8 output o := i.aggregate(over: 1s, using: sum)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].expr_id;
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
        let wref = WRef::SlidingRef(0);
        assert!(matches!(
            expr.kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::SlidingWindow(WRef::SlidingRef(0)), _)
        ));
        let window = &ir.mode.ir_expr_res.windows[&ExprId(0)].clone();
        assert_eq!(
            window,
            &SlidingWindow {
                target: SRef::InRef(0),
                caller: SRef::OutRef(0),
                wait: false,
                reference: wref,
                op: WindowOperation::Sum,
                eid: ExprId(0),
                duration: Duration::from_secs(1)
            }
        );
    }

    #[test]
    fn parameter_expr() {
        let spec = "output o(a,b,c) :=  if c then a else b";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].expr_id;
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::Ite{..}));
        if let ExpressionKind::Ite { condition, consequence, alternative } = &expr.kind {
            assert!(matches!(condition.kind, ExpressionKind::ParameterAccess(_, 2)));
            assert!(matches!(consequence.kind, ExpressionKind::ParameterAccess(_, 0)));
            assert!(matches!(alternative.kind, ExpressionKind::ParameterAccess(_, 1)));
        } else {
            unreachable!()
        }
    }

    #[test]
    fn parametrized_access() {
        use crate::common_ir::StreamAccessKind;
        let spec = "output o(a,b,c) :=  if c then a else b output A := o(1,2,true).offset(by:-1)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[1].expr_id;
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
        assert!(matches!(
            expr.kind,
            ExpressionKind::StreamAccess(_, StreamAccessKind::Offset(Offset::PastDiscreteOffset(_)), _)
        ));
        if let ExpressionKind::StreamAccess(sr, _, v) = &expr.kind {
            assert_eq!(*sr, SRef::OutRef(0));
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
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
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
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::TupleAccess(_, 1)));
    }

    #[test]
    fn simple_trigger() {
        let spec = "trigger true";
        let ir = obtain_expressions(spec);
        assert_eq!(ir.num_triggers(), 1);
        let tr = &ir.triggers[0];
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&tr.expr_id];
        assert!(matches!(expr.kind, ExpressionKind::LoadConstant(_)));
    }

    #[test]
    fn input_trigger() {
        use crate::hir::expression::ArithLogOp;
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
        use crate::hir::expression::ArithLogOp;
        let spec = "output o := 3 + 5 ";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].expr_id;
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
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
            ("input i :(Int8,UInt8)", AnnotatedType::Tuple(vec![AnnotatedType::Int(8), AnnotatedType::UInt(8)])),
            ("input i :Float32?", AnnotatedType::Option(Box::new(AnnotatedType::Float(32)))),
        ] {
            let ir = obtain_expressions(spec);
            let input = &ir.inputs[0];
            let transformed_type = &input.annotated_type;
            assert_eq!(transformed_type, ty);
        }
    }

    #[test]
    fn functions() {
        use crate::common_ir::StreamAccessKind;
        let spec = "import math output o(a: Int) := max(3,4) output c := o(1)";
        let ir = obtain_expressions(spec);
        assert_eq!(ir.mode.ir_expr_res.func_table.len(), 1);
        let output_expr_id = ir.outputs[1].expr_id;
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::StreamAccess(SRef::OutRef(0), StreamAccessKind::Sync, _)));
    }

    #[test]
    fn function_param_default() {
        let spec = "import math output o(a: Int) := sqrt(a) output c := o(1).defaults(to:1)";
        let ir = obtain_expressions(spec);
        assert_eq!(ir.mode.ir_expr_res.func_table.len(), 1);
        let output_expr_id = ir.outputs[1].expr_id;
        let expr = &ir.mode.ir_expr_res.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::Default{
            expr: _,
            default: _
        }));
        if let ExpressionKind::Default { expr: ex, default } = &expr.kind {
            assert!(matches!(default.kind, ExpressionKind::LoadConstant(_)));
            assert!(matches!(ex.kind, ExpressionKind::StreamAccess(SRef::OutRef(0), _, _)));
        } else {
            unreachable!()
        }
    }

    #[test]
    fn test_instance() {
        use crate::hir::{Output, AC};
        let spec = "input i: Bool output c(a: Int8): Bool @1Hz spawn with i if i filter i close i:= i";
        let ir = obtain_expressions(spec);
        let output: Output = ir.outputs[0].clone();
        assert!(output.annotated_type.is_some());
        assert!(matches!(output.activation_condition, Some(AC::Frequency { .. })));
        assert!(output.params.len() == 1);
        let fil: &Expression = ir.filter(output.sr).unwrap();
        assert!(matches!(fil.kind, ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)));
        let close = ir.close(output.sr).unwrap();
        assert!(matches!(close.kind, ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)));
        let (invoke, op_cond) = ir.spawn(output.sr).unwrap();
        assert!(matches!(invoke.unwrap().kind, ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)));
        assert!(matches!(op_cond.unwrap().kind, ExpressionKind::StreamAccess(_, StreamAccessKind::Sync, _)));
    }

    #[test]
    fn test_ac() {
        use crate::hir::{Output, AC};
        let spec = "input in: Bool output a: Bool @2.5Hz := true output b: Bool @in := false";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(matches!(a.activation_condition, Some(AC::Frequency { .. })));
        let b: &Output = &ir.outputs[1];
        assert!(matches!(b.activation_condition, Some(AC::Expr(_))));
    }

    #[test]
    fn test_spawn_missing_exp() {
        use crate::hir::{Output, SpawnTemplate};
        let spec = "output a: Bool @2.5Hz spawn with () if true := true";
        let ir = obtain_expressions(spec);
        let a: Output = ir.outputs[0].clone();
        assert!(matches!(a.instance_template.spawn, Some(SpawnTemplate { target: None, .. })));
    }
}
