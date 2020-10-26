use crate::{
    hir::expression::{
        Constant as HIRConstant, DiscreteWindow, ExprId, Expression, ExpressionKind, SlidingWindow,
        StreamAccessKind as IRAccess,
    },
    hir::{Hir, Window},
};

use super::IrExpression;
use crate::analysis::naming::{Declaration, DeclarationTable, NamingAnalysis};
use crate::ast;
use crate::ast::{Literal, StreamAccessKind};
use crate::common_ir::{Offset, SRef, WRef};
use crate::reporting::Handler;
use crate::FrontendConfig;
use crate::Raw;
use std::collections::HashMap;
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
        self.windows.clone()
    }
    fn expr(&self, sr: SRef) -> &Expression {
        self.expressions.get(&sr).expect("accessing non-existent expression")
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
        self.mode.windows.clone()
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
    ast_expressions: HashMap<SRef, ast::Expression>,
    transformed_expression: HashMap<SRef, Expression>,
    sliding_windows: Vec<SlidingWindow>,
    discrete_windows: Vec<DiscreteWindow>,
    windows: Vec<Window>,
    ident_to_ref: HashMap<String, SRef>,
    decl_table: DeclarationTable,
}

impl ExpressionTransformer {
    fn get_stream_ref(&self, expr: &ast::Expression) -> Result<SRef, TransformationError> {
        if let ast::ExpressionKind::Ident(_) = &expr.kind {
            let decl = self.decl_table[&expr.id].clone();
            match decl {
                Declaration::In(input) => Ok(self.ident_to_ref[&input.name.name]),
                Declaration::Out(out) => Ok(self.ident_to_ref[&out.name.name]),
                Declaration::ParamOut(pout) => Ok(self.ident_to_ref[&pout.name.name]),
                Declaration::Param(_) | Declaration::Const(_) | Declaration::Func(_) | Declaration::Type(_) => {
                    Err(TransformationError::InvalidIdentRef(decl))
                }
            }
        } else {
            Err(TransformationError::InvalidRefExpr(String::from("Non-identifier transformed to SRef")))
        }
    }

    fn next_exp_id(&mut self) -> ExprId {
        todo!()
    }

    fn transform_literal(&self, lit: &Literal) -> HIRConstant {
        match &lit.kind {
            ast::LitKind::Bool(b) => HIRConstant::Bool(*b),
            ast::LitKind::Str(s) | ast::LitKind::RawStr(s) => HIRConstant::Str(s.clone()),
            ast::LitKind::Numeric(num_str, postfix) => {
                assert!(postfix.is_none());
                HIRConstant::Numeric(num_str.clone())
            }
        }
    }

    fn transform_expression(&mut self, ast_expression: ast::Expression) -> Expression {
        let new_id = self.next_exp_id();
        let span = ast_expression.span;
        let kind: ExpressionKind = match ast_expression.kind {
            ast::ExpressionKind::Lit(lit) => {
                let constant = self.transform_literal(&lit);
                ExpressionKind::LoadConstant(constant)
            }
            ast::ExpressionKind::Ident(_) => match &self.decl_table[&ast_expression.id] {
                Declaration::In(_) | Declaration::Out(_) => match self.get_stream_ref(&ast_expression) {
                    Ok(sref) => ExpressionKind::StreamAccess(sref, IRAccess::Sync),
                    Err(_) => todo!(),
                },
                Declaration::Const(cons) => {
                    let cons = &**cons;
                    let constant = self.transform_literal(&cons.literal);
                    ExpressionKind::LoadConstant(constant)
                }
                _ => unreachable!(),
            },
            ast::ExpressionKind::StreamAccess(expr, kind) => {
                let access_kind = if let StreamAccessKind::Hold = kind { IRAccess::Hold } else { IRAccess::Hold }; //TODO
                let expr_ref = self.get_stream_ref(&*expr).unwrap(); //TODO error case
                ExpressionKind::StreamAccess(expr_ref, access_kind)
            }
            ast::ExpressionKind::Default(expr, def) => ExpressionKind::Default {
                expr: self.transform_expression(*expr).into(),
                default: self.transform_expression(*def).into(),
            },
            ast::ExpressionKind::Offset(ref target_expr, offset) => {
                let ir_offset = match offset {
                    ast::Offset::Discrete(i) if i > 0 => Offset::FutureDiscreteOffset(i.abs() as u32),
                    ast::Offset::Discrete(i) => Offset::PastDiscreteOffset(i.abs() as u32),
                    ast::Offset::RealTime(_, _) => {
                        let dur = parse_duration_from_expr(&ast_expression);
                        if dur < Duration::from_secs(0) {
                            Offset::PastRealTimeOffset(dur)
                        } else {
                            Offset::FutureRealTimeOffset(dur)
                        }
                    }
                };
                let expr_ref = self.get_stream_ref(&*target_expr).unwrap(); //TODO error case
                ExpressionKind::StreamAccess(expr_ref, IRAccess::Offset(ir_offset))
            }
            ast::ExpressionKind::SlidingWindowAggregation { expr: w_expr, duration, wait, aggregation: win_op } => {
                if let Ok(sref) = self.get_stream_ref(&w_expr) {
                    let idx = self.sliding_windows.len();
                    let wref = WRef::SlidingRef(idx);
                    let duration = parse_duration_from_expr(&*duration);
                    self.windows.push(Window { expr: new_id });
                    let window =
                        SlidingWindow { target: sref, duration, wait, op: win_op, reference: wref, eid: new_id };
                    self.sliding_windows.push(window);
                    ExpressionKind::Window(WRef::SlidingRef(idx))
                } else {
                    todo!()
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
                let arguments: Vec<Expression> =
                    vec![self.transform_expression(*left), self.transform_expression(*right)];
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
                let arguments: Vec<Expression> = vec![self.transform_expression(*arg)];
                ExpressionKind::ArithLog(arith_op, arguments)
            }
            ast::ExpressionKind::Ite(cond, cons, alt) => {
                let condition = self.transform_expression(*cond).into();
                let consequence = self.transform_expression(*cons).into();
                let alternative = self.transform_expression(*alt).into();
                ExpressionKind::Ite { condition, consequence, alternative }
            }
            ast::ExpressionKind::ParenthesizedExpression(_, inner, _) => {
                return self.transform_expression(*inner);
            }
            ast::ExpressionKind::MissingExpression => unimplemented!(),
            ast::ExpressionKind::Tuple(inner) => {
                ExpressionKind::Tuple(inner.into_iter().map(|ex| self.transform_expression(*ex).into()).collect())
            }
            ast::ExpressionKind::Field(inner_exp, ident) => {
                let num: usize = ident.name.parse().expect("checked in AST verifier");
                let inner = self.transform_expression(*inner_exp).into();
                ExpressionKind::TupleAccess(inner, num)
            }
            ast::ExpressionKind::Method(_base, _name, _types, _params) => todo!(),
            ast::ExpressionKind::Function(name, _type_param, args) => {
                //TODO use type_param
                let _decl = self.decl_table[&ast_expression.id].clone();
                ExpressionKind::Function(
                    name.name.name,
                    args.into_iter().map(|ex| self.transform_expression(*ex).into()).collect(),
                )
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

impl Hir<IrExpression> {
    pub fn transform_expressions(raw: Hir<Raw>, handler: &Handler, config: &FrontendConfig) -> Self {
        let Hir { inputs, outputs, triggers, next_input_ref, next_output_ref, mode: raw_mode } = raw;

        let Raw { constants: _constants, expressions: ast_expressions } = raw_mode;

        let ident_to_ref: HashMap<String, SRef> = inputs
            .iter()
            .enumerate()
            .map(|(n, i)| (i.name.clone(), SRef::InRef(n)))
            .chain(outputs.iter().enumerate().map(|(n, o)| (o.name.clone(), SRef::OutRef(n))))
            .collect();

        let expr_transformer = ExpressionTransformer {
            ast_expressions,
            transformed_expression: HashMap::new(),
            sliding_windows: vec![],
            discrete_windows: vec![],
            windows: vec![],
            ident_to_ref,
            decl_table: Default::default(),
        };

        let _naming_analyzer = NamingAnalysis::new(&handler, *config);
        //let mut decl_table = naming_analyzer.check(&raw, constants);

        let ExpressionTransformer { transformed_expression: result_map, windows, .. } = expr_transformer;

        let new_mode = IrExpression { expressions: result_map, windows };

        Hir { inputs, outputs, triggers, next_input_ref, next_output_ref, mode: new_mode }
    }
}
