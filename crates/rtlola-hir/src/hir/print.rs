use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result};

use itertools::Itertools;

use super::{AnnotatedType, Offset, WindowReference};
use crate::hir::expression::{ArithLogOp, Constant, Expression, Literal};
use crate::hir::{FnExprKind, Inlined, StreamAccessKind, StreamReference, WidenExprKind};

impl Expression {
    /// Produces a prettified string representation of the expression given the names of the streams
    pub(crate) fn pretty_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        use crate::hir::expression::ExpressionKind::*;
        match &self.kind {
            StreamAccess(sref, kind, params) => {
                format!(
                    "{}{}{}",
                    names[&sref],
                    if !params.is_empty() {
                        format!("({})", params.iter().map(|e| e.pretty_string(names)).join(", "))
                    } else {
                        "".into()
                    },
                    match kind {
                        StreamAccessKind::Offset(o) => format!(".offset(by: {})", o),
                        StreamAccessKind::Hold => ".hold()".into(),
                        StreamAccessKind::SlidingWindow(r) | StreamAccessKind::DiscreteWindow(r) => {
                            format!(".aggregate(ref: {})", r)
                        },
                        _ => "".into(),
                    }
                )
            },
            LoadConstant(c) => format!("{}", c),
            Function(FnExprKind { name, args, .. }) => {
                format!("{}({})", name, args.iter().map(|e| e.pretty_string(names)).join(", "))
            },
            Tuple(elems) => format!("({})", elems.iter().map(|e| e.pretty_string(names)).join(", ")),
            Ite {
                condition,
                consequence,
                alternative,
                ..
            } => {
                format!(
                    "if {} then {} else {}",
                    condition.pretty_string(names),
                    consequence.pretty_string(names),
                    alternative.pretty_string(names)
                )
            },
            ArithLog(op, args) => {
                if args.len() == 1 {
                    format!("{}{}", op, args.get(0).unwrap().pretty_string(names))
                } else {
                    format!(
                        "({})",
                        args.iter().map(|e| e.pretty_string(names)).join(&format!(" {} ", op))
                    )
                }
            },
            Default { expr, default } => {
                format!(
                    "{}.default({})",
                    expr.pretty_string(names),
                    default.pretty_string(names)
                )
            },
            Widen(WidenExprKind { expr: e, ty }) => format!("{}({})", ty, e.pretty_string(names)),
            TupleAccess(e, idx) => format!("{}.{}", e.pretty_string(names), idx),
            ParameterAccess(sref, idx) => format!("Param({}, {})", names[&sref], idx),
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use crate::hir::expression::ExpressionKind::*;
        match &self.kind {
            LoadConstant(c) => write!(f, "{}", c),
            Function(FnExprKind { name, args, .. }) => {
                write!(f, "{}({})", name, args.iter().map(|e| format!("{}", e)).join(", "))
            },
            Tuple(elems) => write!(f, "({})", elems.iter().map(|e| format!("{}", e)).join(", ")),
            Ite {
                condition,
                consequence,
                alternative,
                ..
            } => {
                write!(f, "if {} then {} else {}", condition, consequence, alternative)
            },
            ArithLog(op, args) => {
                if args.len() == 1 {
                    write!(f, "{}{}", op, args.get(0).unwrap())
                } else {
                    write!(
                        f,
                        "({})",
                        args.iter().map(|e| format!("{}", e)).join(&format!(" {} ", op))
                    )
                }
            },
            Default { expr, default } => write!(f, "{}.default({})", expr, default),
            Widen(WidenExprKind { expr: e, ty }) => write!(f, "{}({})", ty, e),
            TupleAccess(e, idx) => write!(f, "{}.{}", e, idx),
            ParameterAccess(sref, idx) => write!(f, "Param(ref: {}, idx: {})", sref, idx),
            StreamAccess(sref, kind, params) => {
                write!(
                    f,
                    "Stream(ref: {}, params: ({}))",
                    sref,
                    params.iter().map(|e| format!("{}", e)).join(", ")
                )?;
                match kind {
                    StreamAccessKind::Offset(o) => write!(f, ".offset(by: {})", o),
                    StreamAccessKind::Hold => write!(f, ".hold()"),
                    StreamAccessKind::SlidingWindow(r) | StreamAccessKind::DiscreteWindow(r) => {
                        write!(f, ".aggregate(ref: {})", r)
                    },
                    _ => Ok(()),
                }
            },
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let lit = match self {
            Constant::Inlined(Inlined { lit, .. }) => lit,
            Constant::Basic(c) => c,
        };
        match lit {
            Literal::SInt(v) => write!(f, "{}", v),
            Literal::Integer(v) => write!(f, "{}", v),
            Literal::Float(v) => write!(f, "{}", v),
            Literal::Bool(v) => write!(f, "{}", v),
            Literal::Str(v) => write!(f, "{}", v),
        }
    }
}

impl Display for ArithLogOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use ArithLogOp::*;
        match self {
            Not => write!(f, "!"),
            Neg => write!(f, "~"),
            Add => write!(f, "+"),
            Sub => write!(f, "-"),
            Mul => write!(f, "*"),
            Div => write!(f, "/"),
            Rem => write!(f, "%"),
            Pow => write!(f, "^"),
            And => write!(f, "∧"),
            Or => write!(f, "∨"),
            Eq => write!(f, "="),
            Lt => write!(f, "<"),
            Le => write!(f, "≤"),
            Ne => write!(f, "≠"),
            Ge => write!(f, "≥"),
            Gt => write!(f, ">"),
            BitNot => write!(f, "~"),
            BitAnd => write!(f, "&"),
            BitOr => write!(f, "|"),
            BitXor => write!(f, "^"),
            Shl => write!(f, "<<"),
            Shr => write!(f, ">>"),
        }
    }
}

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Offset::PastDiscrete(u) => write!(f, "{}", u),
            _ => unimplemented!(),
        }
    }
}

impl Display for WindowReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            WindowReference::Sliding(u) => write!(f, "SlidingWin({})", u),
            WindowReference::Discrete(u) => write!(f, "DiscreteWin({})", u),
        }
    }
}

impl Display for StreamReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            StreamReference::Out(ix) => write!(f, "Out({})", ix),
            StreamReference::In(ix) => write!(f, "In({})", ix),
        }
    }
}

impl Display for AnnotatedType {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use AnnotatedType::*;
        match self {
            Int(s) => write!(f, "Int{}", s),
            Float(s) => write!(f, "Float{}", s),
            UInt(s) => write!(f, "UInt{}", s),
            Bool => write!(f, "Bool"),
            String => write!(f, "String"),
            Bytes => write!(f, "Bytes"),
            Option(t) => write!(f, "Option<{}>", t),
            Tuple(tys) => write!(f, "({})", tys.iter().map(|t| format!("{}", t)).join(",")),
            //Used in function declaration
            Numeric => write!(f, "Numeric"),
            Sequence => write!(f, "Sequence"),
            Param(idx, name) => write!(f, "FunctionParam({}, {})", *idx, name),
        }
    }
}
