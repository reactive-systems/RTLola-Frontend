use crate::{
    hir::expression::{
        Constant as HIRConstant, ConstantLiteral, DiscreteWindow, ExprId, Expression, ExpressionKind, SlidingWindow,
        StreamAccessKind as IRAccess,
    },
    hir::{AnnotatedType, Hir, Input, InstanceTemplate, Output, Parameter, SpawnTemplate, Trigger, Window},
};

use super::IrExpression;
use crate::analysis::naming::{Declaration, NamingAnalysis};
use crate::ast;
use crate::ast::{Ast, Literal, StreamAccessKind, Type};
use crate::common_ir::{Offset, SRef, WRef};
use crate::parse::NodeId;
use crate::reporting::Handler;
use crate::stdlib::FuncDecl;
use crate::FrontendConfig;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;

pub(crate) trait WithIrExpr {
    fn windows(&self) -> Vec<Window>;
    fn expr(&self, sr: SRef) -> &Expression;
    fn spawn(&self, sr: SRef) -> (&Expression, &Expression);
    fn filter(&self, sr: SRef) -> &Expression;
    fn close(&self, sr: SRef) -> &Expression;
}

impl WithIrExpr for IrExpression {
    fn windows(&self) -> Vec<Window> {
        todo!()
    }
    fn expr(&self, sr: SRef) -> &Expression {
        match sr {
            SRef::InRef(_) => todo!(),
            SRef::OutRef(_) => todo!(),
        }
    }
    fn spawn(&self, _sr: SRef) -> (&Expression, &Expression) {
        todo!()
    }
    fn filter(&self, _sr: SRef) -> &Expression {
        todo!()
    }
    fn close(&self, _sr: SRef) -> &Expression {
        todo!()
    }
}

impl Hir<IrExpression> {
    fn windows(&self) -> Vec<Window> {
        todo!()
    }
}

pub(crate) trait IrExprWrapper {
    type InnerE: WithIrExpr;
    fn inner_expr(&self) -> &Self::InnerE;
}

impl<A: IrExprWrapper<InnerE = T>, T: WithIrExpr + 'static> WithIrExpr for A {
    fn windows(&self) -> Vec<Window> {
        self.inner_expr().windows()
    }
    fn expr(&self, sr: SRef) -> &Expression {
        self.inner_expr().expr(sr)
    }
    fn spawn(&self, sr: SRef) -> (&Expression, &Expression) {
        self.inner_expr().spawn(sr)
    }
    fn filter(&self, sr: SRef) -> &Expression {
        self.inner_expr().filter(sr)
    }
    fn close(&self, sr: SRef) -> &Expression {
        self.inner_expr().close(sr)
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
            _ => unimplemented!("todo error"),
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
            ast::LitKind::Numeric(num_str, postfix) => ConstantLiteral::Numeric(num_str.clone(), postfix.clone()),
        }
    }

    fn transform_expression(&mut self, ast_expression: ast::Expression, current_output: SRef) -> Expression {
        let new_id = self.next_exp_id();
        let span = ast_expression.span;
        dbg!(&ast_expression.kind);
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
                Declaration::ParamOut(_o) => todo!("may be unreachable"),
                Declaration::Param(p) => ExpressionKind::ParameterAccess(p.param_idx),
                Declaration::Func(_) => todo!("Function identifier transform"),
                Declaration::Type(_) => todo!("type identifier transform"),
            },
            ast::ExpressionKind::StreamAccess(expr, kind) => {
                let access_kind = if let StreamAccessKind::Hold = kind { IRAccess::Hold } else { IRAccess::Sync }; //TODO
                let (expr_ref, args) = self.get_stream_ref(&*expr, current_output).unwrap(); //TODO error case
                                                                                             //TODO param function case
                ExpressionKind::StreamAccess(expr_ref, access_kind, args)
            }
            ast::ExpressionKind::Default(expr, def) => ExpressionKind::Default {
                expr: self.transform_expression(*expr, current_output).into(),
                default: self.transform_expression(*def, current_output).into(),
            },
            ast::ExpressionKind::Offset(ref target_expr, offset) => {
                use uom::si::time::{nanosecond, second};
                let ir_offset = match offset {
                    ast::Offset::Discrete(i) if i > 0 => Offset::FutureDiscreteOffset(i.abs() as u32),
                    ast::Offset::Discrete(i) => Offset::PastDiscreteOffset(i.abs() as u32),
                    ast::Offset::RealTime(_, _) => {
                        let offset_uom_time =
                            offset.to_uom_time().expect("ast::Offset::RealTime should return uom_time");
                        //let dur = offset_uom_time.abs().to_u32().expect("offset to big for u32");
                        //let dur = Duration::from_secs(dur.get::<second>());
                        let dur = offset_uom_time.get::<nanosecond>().to_integer();
                        //assert!(dur.into() > 0);
                        let dur = Duration::from_nanos(dur as u64);
                        if offset_uom_time.get::<second>().numer() > &0i64 {
                            Offset::PastRealTimeOffset(dur) //TODO!
                        } else {
                            Offset::FutureRealTimeOffset(dur) //TODO FIXME
                        }
                    }
                };
                let (expr_ref, args) = self.get_stream_ref(&*target_expr, current_output).unwrap(); //TODO error case
                ExpressionKind::StreamAccess(expr_ref, IRAccess::Offset(ir_offset), args)
            }
            ast::ExpressionKind::DiscreteWindowAggregation { .. } => todo!(),
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
                    ExpressionKind::Window(WRef::SlidingRef(idx))
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
                //TODO use type_param
                //let _decl = self.stream_by_name[&ast_expression.id].clone();
                let decl: Declaration = self.decl_table[&ast_expression.id].clone();
                match decl {
                    Declaration::Func(_) => ExpressionKind::Function {
                        name: name.name.name,
                        args: args.into_iter().map(|ex| self.transform_expression(*ex, current_output)).collect(),
                        type_param: type_param
                            .into_iter()
                            .map(|t| annotated_type(&t).expect("given type arguments have to be replaceable"))
                            .collect(),
                    },
                    Declaration::ParamOut(_) => todo!(),
                    _ => todo!("error case"),
                }
                /*ExpressionKind::Function(
                    name.name.name,
                    args.into_iter().map(|ex| self.transform_expression(*ex).into()).collect(),
                )
                */
            }
        };
        dbg!(&kind);
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
    ts: Option<ast::TemplateSpec>,
    exprid_to_expr: &mut HashMap<ExprId, Expression>,
    current_output: SRef,
) -> InstanceTemplate {
    if let Some(ts) = ts {
        let invoke_spec = if let Some(inv_spec) = ts.inv {
            let is = SpawnTemplate {
                target: {
                    insert_return(exprid_to_expr, transformer.transform_expression(inv_spec.target, current_output))
                },
                condition: inv_spec.condition.map(|cond_expr| {
                    insert_return(exprid_to_expr, transformer.transform_expression(cond_expr, current_output))
                }),
                is_if: inv_spec.is_if,
            };
            Some(is)
        } else {
            None
        };
        InstanceTemplate {
            spawn: invoke_spec,
            filter: ts.ext.map(|ext_spec| {
                insert_return(exprid_to_expr, transformer.transform_expression(ext_spec.target, current_output))
            }),
            close: ts.ter.map(|ter_spec| {
                insert_return(exprid_to_expr, transformer.transform_expression(ter_spec.target, current_output))
            }),
        }
    } else {
        InstanceTemplate { spawn: None, filter: None, close: None }
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
            let ast::Output { expression, name, params, template_spec, ty, extend, .. } =
                //Rc::try_unwrap(o).expect("other strong references should be dropped now");
                (*o).clone();
            let params: Vec<Parameter> = params
                .iter()
                .enumerate()
                .map(|(ix, p)| {
                    assert_eq!(ix, p.param_idx);
                    Parameter { name: p.name.name.clone(), annotated_type: annotated_type(&p.ty), idx: p.param_idx }
                })
                .collect();
            let annotated_type = annotated_type(&ty);
            let expression = expr_transformer.transform_expression(expression, sr);
            let expr_id = expression.eid;
            exprid_to_expr.insert(expr_id, expression);
            let activation_condition = extend
                .expr
                .map(|act| insert_return(&mut exprid_to_expr, expr_transformer.transform_expression(act, sr)));
            let template_spec = transform_template_spec(&mut expr_transformer, template_spec, &mut exprid_to_expr, sr);
            hir_outputs.push(Output {
                name: name.name,
                sr,
                params,
                template_spec,
                annotated_type,
                activation_condition,
                expr_id,
            });
        }
        let hir_outputs = hir_outputs;
        let mut hir_triggers = vec![];
        for (ix, t) in trigger.into_iter().enumerate() {
            let sr = SRef::OutRef(hir_outputs.len() + ix);
            let ast::Trigger { message, name, expression, .. } =
                Rc::try_unwrap(t).expect("other strong references should be dropped now");
            let expr_id = insert_return(&mut exprid_to_expr, expr_transformer.transform_expression(expression, sr));

            hir_triggers.push(Trigger::new(name, message, expr_id, sr));
        }
        let hir_triggers = hir_triggers;
        let hir_inputs: Vec<Input> = inputs
            .into_iter()
            .enumerate()
            .map(|(ix, i)| Input {
                annotated_type: annotated_type(&i.ty).expect("Input Streams must have type annotation"),
                //name: Rc::try_unwrap(i).expect("other strong references should be dropped now").name.name,
                name: i.name.name.clone(),
                sr: SRef::InRef(ix),
            })
            .collect();

        let ExpressionTransformer { sliding_windows, discrete_windows: _, .. } = expr_transformer;

        let windows: HashMap<ExprId, SlidingWindow> = sliding_windows.into_iter().map(|w| (w.eid, w)).collect();

        let new_mode = IrExpression { exprid_to_expr, windows, func_table };

        Hir {
            next_input_ref: hir_inputs.len(),
            inputs: hir_inputs,
            next_output_ref: hir_outputs.len(),
            outputs: hir_outputs,
            triggers: hir_triggers,
            mode: new_mode,
        }

        //let Hir { inputs, outputs, triggers, next_input_ref, next_output_ref, mode: raw_mode } = raw;
        /*
        let Raw { constants, expressions: ast_expressions, template_specs } = raw_mode;

        let ident_to_ref: HashMap<String, SRef> = inputs
            .iter()
            .enumerate()
            .map(|(n, i)| (i.name.clone(), SRef::InRef(n)))
            .chain(outputs.iter().enumerate().map(|(n, o)| (o.name.clone(), SRef::OutRef(n))))
            .collect();

        let (result_map, windows) = expr_transformer.transform();

        let new_mode = IrExpression { expressions: result_map, windows };

        Hir { inputs, outputs, triggers, next_input_ref, next_output_ref, mode: new_mode }
        */
    }
}

#[allow(unused_variables)]
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
            if string.starts_with("Int") {
                let size: u32 = string[3..].parse().expect("");
                return Some(AnnotatedType::Int(size));
            }
            if string.starts_with("UInt") {
                let size: u32 = string[4..].parse().expect("");
                return Some(AnnotatedType::UInt(size));
            }
            if string.starts_with("Float") {
                let size: u32 = string[5..].parse().expect("");
                return Some(AnnotatedType::Float(size));
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
    use crate::parse::{parse, SourceMapper};
    use std::path::PathBuf;

    fn obtain_expressions(spec: &str) -> Hir<IrExpression> {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        dbg!(&ast);
        let replaced: Hir<IrExpression> = Hir::<IrExpression>::transform_expressions(ast, &handler, &config);
        replaced
    }

    #[test]
    fn window_len() {
        let ir = obtain_expressions("output a @1Hz := 1 output b @1min:= a.aggregate(over: 1s, using: sum)");
        assert_eq!(1, ir.mode.windows.len());
        //TODO
    }

    #[test]
    fn all() {
        //Tests all cases are implemented
        let spec = "input i: Int8 \
        output o := 3
        output o2 @1Hz := 4
        output o3 := if true then 1 else 2
        output o4 := 3 + 4
        output o5 := o
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
        let expr = &ir.mode.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::Default{..}));
    }

    #[test]
    fn transform_offset() {
        use crate::hir::expression::StreamAccessKind;
        //TODO do remaining cases
        for (spec, offset) in &[
            ("input o :Int8 output off := o", StreamAccessKind::Sync),
            //("input o :Int8 output off := o.aggregate(over: 1s, using: sum)",StreamAccessKind::DiscreteWindow(WRef::SlidingRef(0))),
            ("input o :Int8 output off := o.hold()", StreamAccessKind::Hold),
            ("input o :Int8 output off := o.offset(by:-1)", StreamAccessKind::Offset(Offset::PastDiscreteOffset(1))),
            ("input o :Int8 output off := o.offset(by: 1)", StreamAccessKind::Offset(Offset::FutureDiscreteOffset(1))),
            //("input o :Int8 output off := o.offset(by:-1s)",StreamAccessKind::Offset(Offset::PastRealTimeOffset(Duration::from_secs(1)))),
            //("input o :Int8 output off := o.offset(by: 1s)",StreamAccessKind::Offset(Offset::FutureRealTimeOffset(Duration::from_secs(1)))),
        ] {
            let ir = obtain_expressions(spec);
            let output_expr_id = ir.outputs[0].expr_id;
            let expr = &ir.mode.exprid_to_expr[&output_expr_id];
            //dbg!(&offset,&expr);
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
        let expr = &ir.mode.exprid_to_expr[&output_expr_id];
        //dbg!(&offset,&expr);
        let wref = WRef::SlidingRef(0);
        assert!(matches!(expr.kind, ExpressionKind::Window(WRef::SlidingRef(0))));
        let window = &ir.mode.windows[&ExprId(0)].clone();
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
        let expr = &ir.mode.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::Ite{..}));
        if let ExpressionKind::Ite { condition, consequence, alternative } = &expr.kind {
            assert!(matches!(condition.kind, ExpressionKind::ParameterAccess(2)));
            assert!(matches!(consequence.kind, ExpressionKind::ParameterAccess(0)));
            assert!(matches!(alternative.kind, ExpressionKind::ParameterAccess(1)));
        } else {
            unreachable!()
        }
    }

    #[test]
    fn parametrized_access() {
        use crate::hir::expression::StreamAccessKind;
        let spec = "output o(a,b,c) :=  if c then a else b output A := o(1,2,true).offset(by:-1)";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[1].expr_id;
        let expr = &ir.mode.exprid_to_expr[&output_expr_id];
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
        let expr = &ir.mode.exprid_to_expr[&output_expr_id];
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
        let expr = &ir.mode.exprid_to_expr[&output_expr_id];
        assert!(matches!(expr.kind, ExpressionKind::TupleAccess(_, 1)));
    }

    #[test]
    fn arith_op() {
        use crate::hir::expression::ArithLogOp;
        let spec = "output o := 3 + 5 ";
        let ir = obtain_expressions(spec);
        let output_expr_id = ir.outputs[0].expr_id;
        let expr = &ir.mode.exprid_to_expr[&output_expr_id];
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
}
