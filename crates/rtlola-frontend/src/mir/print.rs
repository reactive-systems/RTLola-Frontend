use std::fmt::{Display, Formatter, Result};

use super::{FloatTy, IntTy, UIntTy};
use crate::mir::{ArithLogOp, Constant, Expression, ExpressionKind, StreamAccessKind, Type};

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            ExpressionKind::LoadConstant(c) => write!(f, "{}", c),
            ExpressionKind::Function(name, args) => {
                write!(f, "{}(", name)?;
                if let Type::Function(arg_tys, res) = &self.ty {
                    let zipped: Vec<(&Type, &Expression)> = arg_tys.iter().zip(args.iter()).collect();
                    if let Some((last, prefix)) = zipped.split_last() {
                        prefix
                            .iter()
                            .fold(Ok(()), |accu, (t, a)| accu.and_then(|()| write!(f, "{}: {}, ", a, t)))?;
                        write!(f, "{}: {}", last.1, last.0)?;
                    }
                    write!(f, ") -> {}", res)
                } else {
                    unreachable!("The type of a function needs to be a function.")
                }
            },
            ExpressionKind::Convert { expr } => write!(f, "cast<{},{}>({})", expr.ty, self.ty, expr),
            ExpressionKind::Tuple(elems) => write_delim_list(f, elems, "(", ")", ","),
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
                ..
            } => {
                write!(f, "if {} then {} else {}", condition, consequence, alternative)
            },
            ExpressionKind::ArithLog(op, args) => {
                write_delim_list(f, args, &format!("{}(", op), &format!(") : [{}]", self.ty), ",")
            },
            ExpressionKind::Default { expr, default, .. } => write!(f, "{}.default({})", expr, default),
            ExpressionKind::StreamAccess(sr, access, para) => {
                assert!(para.is_empty());
                match access {
                    StreamAccessKind::Sync => write!(f, "{}", sr),
                    StreamAccessKind::Hold => write!(f, "{}.hold()", sr),
                    StreamAccessKind::Offset(offset) => write!(f, "{}.offset({})", sr, offset),
                    StreamAccessKind::SlidingWindow(wr) | StreamAccessKind::DiscreteWindow(wr) => write!(f, "{}", wr),
                }
            },
            ExpressionKind::TupleAccess(expr, num) => write!(f, "{}.{}", expr, num),
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Constant::Bool(b) => write!(f, "{}", b),
            Constant::UInt(u) => write!(f, "{}", u),
            Constant::Int(i) => write!(f, "{}", i),
            Constant::Float(fl) => write!(f, "{}", fl),
            Constant::Str(s) => write!(f, "{}", s),
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

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Type::Float(_) => write!(f, "Float{}", self.size().expect("Floats are sized.").0 * 8),
            Type::UInt(_) => write!(f, "UInt{}", self.size().expect("UInts are sized.").0 * 8),
            Type::Int(_) => write!(f, "Int{}", self.size().expect("Ints are sized.").0 * 8),
            Type::Function(args, res) => write_delim_list(f, args, "(", &format!(") -> {}", res), ","),
            Type::Tuple(elems) => write_delim_list(f, elems, "(", ")", ","),
            Type::String => write!(f, "String"),
            Type::Bytes => write!(f, "Bytes"),
            Type::Option(inner) => write!(f, "Option<{}>", inner),
            Type::Bool => write!(f, "Bool"),
        }
    }
}

impl Display for IntTy {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            IntTy::Int8 => write!(f, "8"),
            IntTy::Int16 => write!(f, "16"),
            IntTy::Int32 => write!(f, "32"),
            IntTy::Int64 => write!(f, "64"),
        }
    }
}

impl Display for UIntTy {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            UIntTy::UInt8 => write!(f, "8"),
            UIntTy::UInt16 => write!(f, "16"),
            UIntTy::UInt32 => write!(f, "32"),
            UIntTy::UInt64 => write!(f, "64"),
        }
    }
}

impl Display for FloatTy {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            FloatTy::Float32 => write!(f, "32"),
            FloatTy::Float64 => write!(f, "64"),
        }
    }
}

/// Writes out the joined vector `v`, enclosed by the given strings `pref` and `suff`.
pub(crate) fn write_delim_list<T: Display>(
    f: &mut Formatter<'_>,
    v: &[T],
    pref: &str,
    suff: &str,
    join: &str,
) -> Result {
    write!(f, "{}", pref)?;
    if let Some(e) = v.first() {
        write!(f, "{}", e)?;
        for b in &v[1..] {
            write!(f, "{}{}", join, b)?;
        }
    }
    write!(f, "{}", suff)?;
    Ok(())
}
