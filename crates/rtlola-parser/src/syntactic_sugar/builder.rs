use rtlola_reporting::Span;

use crate::{
    ast::{
        AnnotatedPacingType, BinOp, EvalSpec, Expression, ExpressionKind, Offset, UnOp,
        WindowOperation,
    },
    RtLolaAst,
};

#[derive(Debug)]
pub(crate) struct Builder<'a> {
    span: Span,
    ast: &'a RtLolaAst,
}

impl<'a> Builder<'a> {
    pub(crate) fn new(span: Span, ast: &'a RtLolaAst) -> Self {
        Self { span, ast }
    }

    pub(crate) fn sync(&self, stream: Expression) -> Expression {
        Expression {
            kind: ExpressionKind::StreamAccess(
                Box::new(stream),
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

    pub(crate) fn discrete_offset(&self, stream: Expression, offset: Expression) -> Expression {
        self.offset(
            stream,
            Offset::Discrete(offset.to_string().parse::<i16>().unwrap()),
        )
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

    pub(crate) fn sub(&self, lhs: Expression, rhs: Expression) -> Expression {
        self.binary(BinOp::Sub, lhs, rhs)
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
