use std::rc::Rc;

use crate::{
    ast::{
        AnnotatedPacingType, CloseSpec, EvalSpec, Expression, ExpressionKind, FunctionName, Ident,
        InstanceSelection, LambdaExpr, Literal, Output, OutputKind, Parameter, SpawnSpec, Tag,
        Type, TypeKind,
    },
    RtLolaAst,
};

impl Parameter {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            name: self.name.next_id(ast),
            ty: self.ty.as_ref().map(|ty| ty.next_id(ast)),
            param_idx: self.param_idx,
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}

impl Type {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            kind: self.kind.next_id(ast),
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}

impl TypeKind {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        match self {
            TypeKind::Simple(s) => TypeKind::Simple(s.clone()),
            TypeKind::Tuple(items) => {
                TypeKind::Tuple(items.iter().map(|ty| ty.next_id(ast)).collect())
            }
            TypeKind::Optional(ty) => TypeKind::Optional(Box::new(ty.next_id(ast))),
        }
    }
}

impl Expression {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            kind: self.kind.next_id(ast),
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}

impl ExpressionKind {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        match self {
            ExpressionKind::Lit(literal) => ExpressionKind::Lit(literal.next_id(ast)),
            ExpressionKind::Ident(ident) => ExpressionKind::Ident(ident.next_id(ast)),
            ExpressionKind::StreamAccess(expression, stream_access_kind) => {
                ExpressionKind::StreamAccess(Box::new(expression.next_id(ast)), *stream_access_kind)
            }
            ExpressionKind::Default(expression, dft) => ExpressionKind::Default(
                Box::new(expression.next_id(ast)),
                Box::new(dft.next_id(ast)),
            ),
            ExpressionKind::Offset(expression, offset) => {
                ExpressionKind::Offset(Box::new(expression.next_id(ast)), *offset)
            }
            ExpressionKind::DiscreteWindowAggregation {
                expr,
                duration,
                wait,
                aggregation,
            } => ExpressionKind::DiscreteWindowAggregation {
                expr: Box::new(expr.next_id(ast)),
                duration: Box::new(duration.next_id(ast)),
                wait: *wait,
                aggregation: *aggregation,
            },
            ExpressionKind::SlidingWindowAggregation {
                expr,
                duration,
                wait,
                aggregation,
            } => ExpressionKind::SlidingWindowAggregation {
                expr: Box::new(expr.next_id(ast)),
                duration: Box::new(duration.next_id(ast)),
                wait: *wait,
                aggregation: *aggregation,
            },
            ExpressionKind::InstanceAggregation {
                expr,
                selection,
                aggregation,
            } => ExpressionKind::InstanceAggregation {
                expr: Box::new(expr.next_id(ast)),
                selection: selection.next_id(ast),
                aggregation: *aggregation,
            },
            ExpressionKind::Binary(bin_op, lhs, rhs) => ExpressionKind::Binary(
                *bin_op,
                Box::new(lhs.next_id(ast)),
                Box::new(rhs.next_id(ast)),
            ),
            ExpressionKind::Unary(un_op, expr) => {
                ExpressionKind::Unary(*un_op, Box::new(expr.next_id(ast)))
            }
            ExpressionKind::Ite(con, cons, alt) => ExpressionKind::Ite(
                Box::new(con.next_id(ast)),
                Box::new(cons.next_id(ast)),
                Box::new(alt.next_id(ast)),
            ),
            ExpressionKind::ParenthesizedExpression(expr) => {
                ExpressionKind::ParenthesizedExpression(Box::new(expr.next_id(ast)))
            }
            ExpressionKind::MissingExpression => ExpressionKind::MissingExpression,
            ExpressionKind::Tuple(exprs) => {
                ExpressionKind::Tuple(exprs.iter().map(|expr| expr.next_id(ast)).collect())
            }
            ExpressionKind::Field(expr, ident) => {
                ExpressionKind::Field(Box::new(expr.next_id(ast)), ident.next_id(ast))
            }
            ExpressionKind::Method(expr, function_name, items, exprs) => ExpressionKind::Method(
                Box::new(expr.next_id(ast)),
                function_name.next_id(ast),
                items.iter().map(|ty| ty.next_id(ast)).collect(),
                exprs.iter().map(|expr| expr.next_id(ast)).collect(),
            ),
            ExpressionKind::Function(function_name, items, exprs) => ExpressionKind::Function(
                function_name.next_id(ast),
                items.iter().map(|ty| ty.next_id(ast)).collect(),
                exprs.iter().map(|expr| expr.next_id(ast)).collect(),
            ),
            ExpressionKind::Lambda(expr) => ExpressionKind::Lambda(expr.next_id(ast)),
        }
    }
}

impl Literal {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            kind: self.kind.clone(),
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}

impl FunctionName {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            name: self.name.next_id(ast),
            arg_names: self
                .arg_names
                .iter()
                .map(|arg| arg.as_ref().map(|arg| arg.next_id(ast)))
                .collect(),
        }
    }
}

impl Ident {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, _ast: &RtLolaAst) -> Self {
        Self {
            name: self.name.clone(),
            span: self.span.to_indirect(),
        }
    }
}

impl LambdaExpr {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            parameters: self
                .parameters
                .iter()
                .map(|p| Rc::new(p.next_id(ast)))
                .collect(),
            expr: Box::new(self.expr.next_id(ast)),
        }
    }
}

impl InstanceSelection {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        match self {
            InstanceSelection::Fresh => InstanceSelection::Fresh,
            InstanceSelection::All => InstanceSelection::All,
            InstanceSelection::FilteredFresh(lambda_expr) => {
                InstanceSelection::FilteredFresh(lambda_expr.next_id(ast))
            }
            InstanceSelection::FilteredAll(lambda_expr) => {
                InstanceSelection::FilteredAll(lambda_expr.next_id(ast))
            }
        }
    }
}

impl AnnotatedPacingType {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        match self {
            AnnotatedPacingType::NotAnnotated(span) => {
                AnnotatedPacingType::NotAnnotated(span.to_indirect())
            }
            AnnotatedPacingType::Global(expr) => AnnotatedPacingType::Global(expr.next_id(ast)),
            AnnotatedPacingType::Local(expr) => AnnotatedPacingType::Local(expr.next_id(ast)),
            AnnotatedPacingType::Unspecified(expr) => {
                AnnotatedPacingType::Unspecified(expr.next_id(ast))
            }
        }
    }
}

impl EvalSpec {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            annotated_pacing: self.annotated_pacing.next_id(ast),
            condition: self.condition.as_ref().map(|c| c.next_id(ast)),
            eval_expression: self.eval_expression.as_ref().map(|expr| expr.next_id(ast)),
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}

impl SpawnSpec {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            expression: self.expression.as_ref().map(|e| e.next_id(ast)),
            annotated_pacing: self.annotated_pacing.next_id(ast),
            condition: self.condition.as_ref().map(|c| c.next_id(ast)),
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}

impl CloseSpec {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            condition: self.condition.next_id(ast),
            annotated_pacing: self.annotated_pacing.next_id(ast),
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}

impl Tag {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, _ast: &RtLolaAst) -> Self {
        Self {
            key: self.key.clone(),
            value: self.value.clone(),
            span: self.span.to_indirect(),
        }
    }
}

impl OutputKind {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        match self {
            OutputKind::NamedOutput(ident) => OutputKind::NamedOutput(ident.next_id(ast)),
            OutputKind::Trigger => OutputKind::Trigger,
        }
    }
}

impl Output {
    /// Creates a copy with a new id and sets the span to indirect
    pub(crate) fn next_id(&self, ast: &RtLolaAst) -> Self {
        Self {
            kind: self.kind.next_id(ast),
            annotated_type: self.annotated_type.as_ref().map(|ty| ty.next_id(ast)),
            params: self
                .params
                .iter()
                .map(|p| Rc::new(p.next_id(ast)))
                .collect(),
            spawn: self.spawn.as_ref().map(|s| s.next_id(ast)),
            eval: self.eval.iter().map(|eval| eval.next_id(ast)).collect(),
            close: self.close.as_ref().map(|close| close.next_id(ast)),
            tags: self.tags.iter().map(|tag| tag.next_id(ast)).collect(),
            id: ast.next_id(),
            span: self.span.to_indirect(),
        }
    }
}
