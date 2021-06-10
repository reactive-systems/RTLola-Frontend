//! This module contains `Display` implementations for the AST.

use std::fmt::{Display, Formatter, Result};

use crate::ast::*;

/// Writes out the joined vector `v`, enclosed by the given strings `pref` and `suff`.
fn write_delim_list<T: Display>(f: &mut Formatter<'_>, v: &[T], pref: &str, suff: &str, join: &str) -> Result {
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

/// Helper to format an Optional
fn format_opt<T: Display>(opt: &Option<T>, pref: &str, suff: &str) -> String {
    if let Some(ref e) = opt {
        format!("{}{}{}", pref, e, suff)
    } else {
        String::new()
    }
}

/// Formats an optional type
fn format_type(ty: &Option<Type>) -> String {
    format_opt(ty, ": ", "")
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "constant {}{} := {}", self.name, format_type(&self.ty), self.literal)
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "input {}", self.name)?;
        if !self.params.is_empty() {
            write_delim_list(f, &self.params, " (", ")", ", ")?;
        }
        write!(f, ": {}", self.ty)
    }
}

impl Display for Output {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "output {}", self.name)?;
        if !self.params.is_empty() {
            write_delim_list(f, &self.params, " (", ")", ", ")?;
        }
        if let Some(ty) = &self.annotated_type {
            write!(f, ": {}", ty)?;
        }
        match &self.annotated_pacing_type {
            None => {},
            Some(pt) => {
                write!(f, " @ {}", pt)?;
            },
        }
        if let Some(spawn) = &self.spawn {
            write!(f, " {}", spawn)?;
        }
        if let Some(filter) = &self.filter {
            write!(f, " {}", filter)?;
        }
        if let Some(close) = &self.close {
            write!(f, " {}", close)?;
        }
        write!(f, " := {}", self.expression)
    }
}

impl Display for Parameter {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.ty {
            None => write!(f, "{}", self.name),
            Some(ty) => write!(f, "{}: {}", self.name, ty),
        }
    }
}

impl Display for SpawnSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if self.target.is_some() || self.condition.is_some() {
            write!(f, "spawn")?;
        }
        if let Some(pt) = &self.annotated_pacing {
            write!(f, " @{}", pt)?;
        }
        if let Some(target) = &self.target {
            write!(f, " with {}", target)?;
        }
        if let Some(condition) = &self.condition {
            write!(f, " {} {}", if self.is_if { "if" } else { "unless" }, condition)?;
        }
        Ok(())
    }
}

impl Display for FilterSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "filter {}", &self.target)
    }
}

impl Display for CloseSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "close {}", self.target)
    }
}

impl Display for Trigger {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "trigger{} {}{}",
            format_opt(&self.annotated_pacing_type, " @", ""),
            self.expression,
            format_opt(&self.message, " \"", "\""),
        )?;
        if !self.info_streams.is_empty() {
            write_delim_list(f, &self.info_streams, " (", ")", ", ")?;
        }
        Ok(())
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.kind)
    }
}

impl Display for TypeKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self {
            TypeKind::Simple(name) => write!(f, "{}", name),
            TypeKind::Tuple(types) => write_delim_list(f, types, "(", ")", ", "),
            TypeKind::Optional(ty) => write!(f, "{}?", ty),
        }
    }
}

impl Display for TypeDeclField {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}: {}", &self.name, &self.ty)
    }
}

impl Display for TypeDeclaration {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "type {}", format_opt(&self.name, "", ""))?;
        write_delim_list(f, &self.fields, " { ", " }", ", ")
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            ExpressionKind::Lit(l) => write!(f, "{}", l),
            ExpressionKind::Ident(ident) => write!(f, "{}", ident),
            ExpressionKind::StreamAccess(expr, access) => {
                match access {
                    StreamAccessKind::Sync => write!(f, "{}", expr),
                    StreamAccessKind::Hold => write!(f, "{}.hold()", expr),
                    StreamAccessKind::Optional => write!(f, "{}.get()", expr),
                }
            },
            ExpressionKind::Default(expr, val) => write!(f, "{}.defaults(to: {})", expr, val),
            ExpressionKind::Offset(expr, val) => write!(f, "{}.offset(by: {})", expr, val),
            ExpressionKind::DiscreteWindowAggregation {
                expr,
                duration,
                wait,
                aggregation,
            } => {
                match wait {
                    true => {
                        write!(
                            f,
                            "{}.aggregate(over_exactly_discrete: {}, using: {})",
                            expr, duration, aggregation
                        )
                    },
                    false => {
                        write!(
                            f,
                            "{}.aggregate(over_discrete: {}, using: {})",
                            expr, duration, aggregation
                        )
                    },
                }
            },
            ExpressionKind::SlidingWindowAggregation {
                expr,
                duration,
                wait,
                aggregation,
            } => {
                match wait {
                    true => {
                        write!(
                            f,
                            "{}.aggregate(over_exactly: {}, using: {})",
                            expr, duration, aggregation
                        )
                    },
                    false => write!(f, "{}.aggregate(over: {}, using: {})", expr, duration, aggregation),
                }
            },
            ExpressionKind::Binary(op, lhs, rhs) => write!(f, "{} {} {}", lhs, op, rhs),
            ExpressionKind::Unary(operator, operand) => write!(f, "{}{}", operator, operand),
            ExpressionKind::Ite(cond, cons, alt) => {
                write!(f, "if {} then {} else {}", cond, cons, alt)
            },
            ExpressionKind::ParenthesizedExpression(left, expr, right) => {
                write!(
                    f,
                    "{}{}{}",
                    if left.is_some() { "(" } else { "" },
                    expr,
                    if right.is_some() { ")" } else { "" }
                )
            },
            ExpressionKind::MissingExpression => Ok(()),
            ExpressionKind::Tuple(exprs) => write_delim_list(f, exprs, "(", ")", ", "),
            ExpressionKind::Function(name, types, args) => {
                write!(f, "{}", name.name)?;
                if !types.is_empty() {
                    write_delim_list(f, types, "<", ">", ", ")?;
                }
                let args: Vec<String> = args
                    .iter()
                    .zip(&name.arg_names)
                    .map(|(arg, arg_name)| {
                        match arg_name {
                            None => format!("{}", arg),
                            Some(arg_name) => format!("{}: {}", arg_name, arg),
                        }
                    })
                    .collect();
                write_delim_list(f, &args, "(", ")", ", ")
            },
            ExpressionKind::Field(expr, ident) => write!(f, "{}.{}", expr, ident),
            ExpressionKind::Method(expr, name, types, args) => {
                write!(f, "{}.{}", expr, name.name)?;
                if !types.is_empty() {
                    write_delim_list(f, types, "<", ">", ", ")?;
                }
                let args: Vec<String> = args
                    .iter()
                    .zip(&name.arg_names)
                    .map(|(arg, arg_name)| {
                        match arg_name {
                            None => format!("{}", arg),
                            Some(arg_name) => format!("{}: {}", arg_name, arg),
                        }
                    })
                    .collect();
                write_delim_list(f, &args, "(", ")", ", ")
            },
        }
    }
}

impl Display for FunctionName {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name)?;
        let args: Vec<String> = self
            .arg_names
            .iter()
            .map(|arg_name| {
                match arg_name {
                    None => String::from("_:"),
                    Some(arg_name) => format!("{}:", arg_name),
                }
            })
            .collect();
        write_delim_list(f, &args, "(", ")", "")
    }
}

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Offset::Discrete(val) => write!(f, "{}", val),
            Offset::RealTime(val, unit) => write!(f, "{}{}", val, unit),
        }
    }
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{}",
            match self {
                TimeUnit::Nanosecond => "ns",
                TimeUnit::Microsecond => "μs",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Second => "s",
                TimeUnit::Minute => "min",
                TimeUnit::Hour => "h",
                TimeUnit::Day => "d",
                TimeUnit::Week => "w",
                TimeUnit::Year => "a",
            }
        )
    }
}

impl Display for WindowOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{}",
            match self {
                WindowOperation::Sum => "Σ",
                WindowOperation::Product => "Π",
                WindowOperation::Average => "avg",
                WindowOperation::Count => "#",
                WindowOperation::Integral => "∫",
                WindowOperation::Min => "min",
                WindowOperation::Max => "max",
                WindowOperation::Disjunction => "∃",
                WindowOperation::Conjunction => "∀",
            }
        )
    }
}

impl Display for UnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{}",
            match self {
                UnOp::Not => "!",
                UnOp::Neg => "-",
                UnOp::BitNot => "~",
            }
        )
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            LitKind::Bool(val) => write!(f, "{}", val),
            LitKind::Numeric(val, unit) => write!(f, "{}{}", val, unit.clone().unwrap_or_default()),
            LitKind::Str(s) => write!(f, "\"{}\"", s),
            LitKind::RawStr(s) => {
                // need to determine padding with `#`
                let mut padding = 0;
                while s.contains(&format!("{}\"", "#".repeat(padding))) {
                    padding += 1;
                }
                write!(f, "r{pad}\"{}\"{pad}", s, pad = "#".repeat(padding))
            },
        }
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name)
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use BinOp::*;
        match &self {
            // Arithmetic
            Add => write!(f, "+"),
            Sub => write!(f, "-"),
            Mul => write!(f, "*"),
            Div => write!(f, "/"),
            Rem => write!(f, "%"),
            Pow => write!(f, "**"),
            And => write!(f, "∧"),
            // Logical
            Or => write!(f, "∨"),
            Eq => write!(f, "="),
            // Comparison
            Lt => write!(f, "<"),
            Le => write!(f, "≤"),
            Ne => write!(f, "≠"),
            Gt => write!(f, ">"),
            Ge => write!(f, "≥"),
            // Bitwise
            BitAnd => write!(f, "&"),
            BitOr => write!(f, "|"),
            BitXor => write!(f, "^"),
            Shl => write!(f, "<<"),
            Shr => write!(f, ">>"),
        }
    }
}

impl Display for Import {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "import {}", self.name)
    }
}

impl Display for RtLolaAst {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for import in &self.imports {
            writeln!(f, "{}", import)?;
        }
        for decl in &self.type_declarations {
            writeln!(f, "{}", decl)?;
        }
        for constant in &self.constants {
            writeln!(f, "{}", constant)?;
        }
        for input in &self.inputs {
            writeln!(f, "{}", input)?;
        }
        for output in &self.outputs {
            writeln!(f, "{}", output)?;
        }
        for trigger in &self.trigger {
            writeln!(f, "{}", trigger)?;
        }
        Ok(())
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.0)
    }
}
