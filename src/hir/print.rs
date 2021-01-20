use crate::common_ir::StreamAccessKind;
use crate::hir::expression::{ArithLogOp, Constant, ConstantLiteral, Expression};
use itertools::Itertools;
use std::fmt::{Display, Formatter, Result};

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use crate::hir::expression::ExpressionKind::*;
        match &self.kind {
            LoadConstant(c) => write!(f, "{}", c),
            Function { name, args, .. } => write!(f, "{}({})", name, args.iter().map(|e| format!("{}", e)).join(", ")),
            Tuple(elems) => write!(f, "({})", elems.iter().map(|e| format!("{}", e)).join(", ")),
            Ite { condition, consequence, alternative, .. } => {
                write!(f, "if {} then {} else {}", condition, consequence, alternative)
            }
            ArithLog(op, args) => {
                if args.len() == 1 {
                    write!(f, "{}{}", op, args.get(0).unwrap())
                } else {
                    write!(f, "({})", args.iter().map(|e| format!("{}", e)).join(&format!(" {} ", op)))
                }
            }
            Default { expr, default } => write!(f, "{}.default({})", expr, default),
            Widen(e, ty) => write!(f, "{}({})", ty, e),
            TupleAccess(e, idx) => write!(f, "{}.{}", e, idx),
            ParameterAccess(sref, idx) => write!(f, "Param(ref: {}, idx: {})", sref, idx),
            StreamAccess(sref, kind, params) => {
                write!(f, "Stream(ref: {}, params: ({}))", sref, params.iter().map(|e| format!("{}", e)).join(", "))?;
                match kind {
                    StreamAccessKind::Offset(o) => write!(f, ".offset(by: {})", o),
                    StreamAccessKind::Hold => write!(f, ".hold()"),
                    StreamAccessKind::SlidingWindow(r) | StreamAccessKind::DiscreteWindow(r) => {
                        write!(f, ".aggregate(ref: {})", r)
                    }
                    _ => Ok(()),
                }
            }
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let lit = match self {
            Constant::InlinedConstant(c, _) => c,
            Constant::BasicConstant(c) => c,
        };
        match lit {
            ConstantLiteral::SInt(v) => write!(f, "{}", v),
            ConstantLiteral::Integer(v) => write!(f, "{}", v),
            ConstantLiteral::Float(v) => write!(f, "{}", v),
            ConstantLiteral::Bool(v) => write!(f, "{}", v),
            ConstantLiteral::Str(v) => write!(f, "{}", v),
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
