#![allow(dead_code)]

use rtlola_reporting::Span;

use crate::{
    ast::{
        AnnotatedPacingType, BinOp, EvalSpec, Expression, ExpressionKind, FunctionName, Ident,
        InstanceOperation, InstanceSelection, Literal, Offset, Type, UnOp, WindowOperation,
    },
    RtLolaAst,
};

#[derive(Debug)]
pub(crate) struct Builder<'a> {
    pub(crate) span: Span,
    pub(crate) ast: &'a RtLolaAst,
}

impl<'a> Builder<'a> {
    pub(crate) fn new(span: Span, ast: &'a RtLolaAst) -> Self {
        Self { span, ast }
    }

    pub(crate) fn ident(&self, ident: Ident) -> Expression {
        Expression {
            kind: ExpressionKind::Ident(ident),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn function(
        &self,
        func_name: FunctionName,
        ty: Vec<Type>,
        parameters: Vec<Expression>,
    ) -> Expression {
        Expression {
            kind: ExpressionKind::Function(func_name, ty, parameters),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn method(
        &self,
        base: Expression,
        func_name: FunctionName,
        ty: Vec<Type>,
        parameters: Vec<Expression>,
    ) -> Expression {
        Expression {
            kind: ExpressionKind::Method(Box::new(base), func_name, ty, parameters),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn last(
        &self,
        stream: Ident,
        params: Vec<Expression>,
        or: Expression,
    ) -> Expression {
        let stream_expr = self.parameter_use(stream, params);
        self.method(
            stream_expr,
            FunctionName {
                name: Ident::new("last".into(), self.span.to_indirect()),
                arg_names: vec![Some(Ident::new("or".into(), self.span.to_indirect()))],
            },
            Vec::new(),
            vec![or],
        )
    }

    pub(crate) fn parameter_use(&self, stream: Ident, params: Vec<Expression>) -> Expression {
        if params.is_empty() {
            self.ident(stream)
        } else {
            self.function(
                FunctionName {
                    name: stream,
                    arg_names: vec![None; params.len()],
                },
                vec![],
                params,
            )
        }
    }

    pub(crate) fn sync(&self, stream: Ident, params: Vec<Expression>) -> Expression {
        Expression {
            kind: ExpressionKind::StreamAccess(
                Box::new(self.parameter_use(stream, params)),
                crate::ast::StreamAccessKind::Sync,
            ),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn offset(&self, stream: Expression, offset: Offset) -> Expression {
        Expression {
            kind: ExpressionKind::Offset(Box::new(stream), offset),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn default(&self, expr: Expression, dft: Expression) -> Expression {
        Expression {
            kind: ExpressionKind::Default(Box::new(expr), Box::new(dft)),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    fn unary(&self, op: UnOp, expr: Expression) -> Expression {
        Expression {
            kind: ExpressionKind::Unary(op, Box::new(expr)),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn not(&self, expr: Expression) -> Expression {
        self.unary(UnOp::Not, expr)
    }

    pub(crate) fn parentesized(&self, expr: Expression) -> Expression {
        Expression {
            kind: ExpressionKind::ParenthesizedExpression(Box::new(expr)),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    fn binary(&self, op: BinOp, lhs: Expression, rhs: Expression) -> Expression {
        Expression {
            kind: ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn or(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::Or, lhs, rhs)
    }

    pub(crate) fn and(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::And, lhs, rhs)
    }

    pub(crate) fn eq(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::Eq, lhs, rhs)
    }

    pub(crate) fn add(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::Add, lhs, rhs)
    }

    pub(crate) fn sub(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::Sub, lhs, rhs)
    }

    pub(crate) fn mul(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::Mul, lhs, rhs)
    }

    pub(crate) fn div(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::Div, lhs, rhs)
    }

    pub(crate) fn sliding_window(
        &self,
        stream: Expression,
        duration: Expression,
        wait: bool,
        aggregation: WindowOperation,
    ) -> Expression {
        Expression {
            kind: ExpressionKind::SlidingWindowAggregation {
                expr: Box::new(stream),
                duration: Box::new(duration),
                wait,
                aggregation,
            },
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn instance_aggregation(
        &self,
        stream: Expression,
        selection: InstanceSelection,
        aggregation: InstanceOperation,
    ) -> Expression {
        Expression {
            kind: ExpressionKind::InstanceAggregation {
                expr: Box::new(stream),
                selection,
                aggregation,
            },
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn all_aggregation(
        &self,
        stream: Expression,
        aggregation: WindowOperation,
    ) -> Expression {
        Expression {
            kind: ExpressionKind::AllAggregation {
                expr: Box::new(stream),
                aggregation,
            },
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn literal(&self, v: Literal) -> Expression {
        Expression {
            kind: ExpressionKind::Lit(v),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn if_then_else(
        &self,
        cond: Expression,
        cons: Expression,
        alt: Expression,
    ) -> Expression {
        Expression {
            kind: ExpressionKind::Ite(Box::new(cond), Box::new(cons), Box::new(alt)),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn tuple(&self, exprs: Vec<Expression>) -> Expression {
        Expression {
            kind: ExpressionKind::Tuple(exprs.into_iter().map(|e| e.next_id(self.ast)).collect()),
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }

    pub(crate) fn eval_spec(
        &self,
        condition: Option<Expression>,
        annotated_pacing: AnnotatedPacingType,
        eval_expression: Option<Expression>,
    ) -> EvalSpec {
        EvalSpec {
            annotated_pacing,
            condition,
            eval_expression,
            id: self.ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}
