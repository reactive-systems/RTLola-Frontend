use std::fmt::{Display, Formatter, Result};

use itertools::Itertools;
use rtlola_hir::hir::OutputKind;

use super::{
    FloatTy, InputStream, InstanceSelection, IntTy, Mir, OutputStream, PacingType, Trigger, UIntTy, Window,
    WindowOperation,
};
use crate::mir::{
    ActivationCondition, ArithLogOp, Constant, Expression, ExpressionKind, Offset, StreamAccessKind, Type,
};

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Constant::Bool(b) => write!(f, "{b}"),
            Constant::UInt(u) => write!(f, "{u}"),
            Constant::Int(i) => write!(f, "{i}"),
            Constant::Float(fl) => write!(f, "{fl:?}"),
            Constant::Str(s) => write!(f, "\"{s}\""),
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
            Type::Function { args, ret } => write_delim_list(f, args, "(", &format!(") -> {ret}"), ","),
            Type::Tuple(elems) => write_delim_list(f, elems, "(", ")", ","),
            Type::String => write!(f, "String"),
            Type::Bytes => write!(f, "Bytes"),
            Type::Option(inner) => write!(f, "Option<{inner}>"),
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

impl Display for InstanceSelection {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            InstanceSelection::Fresh => write!(f, "Fresh"),
            InstanceSelection::All => write!(f, "All"),
        }
    }
}

/// Writes out the joined vector `v`, enclosed by the given strings `pref` and `suff`.
/// Uses the formatter.
pub(crate) fn write_delim_list<T: Display>(
    f: &mut Formatter<'_>,
    v: &[T],
    pref: &str,
    suff: &str,
    join: &str,
) -> Result {
    write!(f, "{pref}")?;
    if let Some(e) = v.first() {
        write!(f, "{e}")?;
        for b in &v[1..] {
            write!(f, "{join}{b}")?;
        }
    }
    write!(f, "{suff}")?;
    Ok(())
}

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Offset::Past(u) => write!(f, "{u}"),
            _ => unimplemented!(),
        }
    }
}

impl Display for WindowOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{}",
            match self {
                WindowOperation::Count => "count",
                WindowOperation::Min => "min",
                WindowOperation::Max => "max",
                WindowOperation::Sum => "sum",
                WindowOperation::Product => "product",
                WindowOperation::Average => "average",
                WindowOperation::Integral => "integral",
                WindowOperation::Conjunction => "conjunction",
                WindowOperation::Disjunction => "disjunction",
                WindowOperation::Last => "last",
                WindowOperation::Variance => "variance",
                WindowOperation::Covariance => "covariance",
                WindowOperation::StandardDeviation => "standard deviation",
                WindowOperation::NthPercentile(_) => todo!(),
            }
        )
    }
}

/// A lightweight wrapper around the Mir to provide a [Display] implementation for Mir struct `T`.
#[derive(Debug, Clone, Copy)]
pub struct RtLolaMirPrinter<'a, T> {
    mir: &'a Mir,
    inner: &'a T,
}

impl<'a, T> RtLolaMirPrinter<'a, T> {
    pub(crate) fn new(mir: &'a Mir, target: &'a T) -> Self {
        RtLolaMirPrinter { mir, inner: target }
    }
}

impl<'a, T: Display> Display for RtLolaMirPrinter<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.inner.fmt(f)
    }
}

impl<'a> Display for RtLolaMirPrinter<'a, ActivationCondition> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            ActivationCondition::Conjunction(s) => {
                let rs = s
                    .iter()
                    .map(|ac| RtLolaMirPrinter::new(self.mir, ac).to_string())
                    .join(&ArithLogOp::And.to_string());
                write!(f, "{rs}")
            },
            ActivationCondition::Disjunction(s) => {
                let rs = s
                    .iter()
                    .map(|ac| RtLolaMirPrinter::new(self.mir, ac).to_string())
                    .join(&ArithLogOp::Or.to_string());
                write!(f, "{rs}")
            },
            ActivationCondition::Stream(s) => write!(f, "{}", self.mir.stream(*s).name()),
            ActivationCondition::True => write!(f, "true"),
        }
    }
}

impl<'a> Display for RtLolaMirPrinter<'a, PacingType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.inner {
            super::PacingType::Periodic(freq) => {
                let s = freq
                    .into_format_args(uom::si::frequency::hertz, uom::fmt::DisplayStyle::Abbreviation)
                    .to_string();
                write!(f, "{}Hz", &s[..s.len() - 3])
            },
            super::PacingType::Event(ac) => RtLolaMirPrinter::new(self.mir, ac).fmt(f),
            super::PacingType::Constant => write!(f, "true"),
        }
    }
}

type Associativity = bool;

fn precedence_level(op: &ArithLogOp) -> (u32, Associativity) {
    // https://en.cppreference.com/w/c/language/operator_precedence
    let precedence = match op {
        ArithLogOp::Not | ArithLogOp::BitNot | ArithLogOp::Neg => 2,

        ArithLogOp::Mul | ArithLogOp::Rem | ArithLogOp::Pow | ArithLogOp::Div => 3,

        ArithLogOp::Add | ArithLogOp::Sub => 4,

        ArithLogOp::Shl | ArithLogOp::Shr => 5,

        ArithLogOp::Lt | ArithLogOp::Le | ArithLogOp::Ge | ArithLogOp::Gt => 6,

        ArithLogOp::Eq | ArithLogOp::Ne => 7,

        ArithLogOp::BitAnd => 8,
        ArithLogOp::BitXor => 9,
        ArithLogOp::BitOr => 10,
        ArithLogOp::And => 11,
        ArithLogOp::Or => 12,
    };

    let associativity = !matches!(op, ArithLogOp::Div | ArithLogOp::Sub);

    (precedence, associativity)
}

pub(crate) fn display_expression(mir: &Mir, expr: &Expression, current_level: u32) -> String {
    match &expr.kind {
        ExpressionKind::LoadConstant(c) => c.to_string(),
        ExpressionKind::ArithLog(op, exprs) => {
            let (op_level, associative) = precedence_level(op);
            let display_exprs = exprs
                .iter()
                .map(|expr| display_expression(mir, expr, op_level))
                .collect::<Vec<_>>();
            let display = match display_exprs.len() {
                1 => format!("{}{}", op, display_exprs[0]),
                2 => format!("{} {} {}", display_exprs[0], op, display_exprs[1]),
                _ => unreachable!(),
            };
            if (associative && current_level < op_level || !associative && current_level <= op_level)
                && current_level != 0
            {
                format!("({display})")
            } else {
                display
            }
        },
        ExpressionKind::StreamAccess {
            target,
            parameters,
            access_kind,
        } => {
            let stream_name = mir.stream(*target).name();
            let target_name = if !parameters.is_empty() {
                let parameter_list = parameters
                    .iter()
                    .map(|parameter| display_expression(mir, parameter, 0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{stream_name}({parameter_list})")
            } else {
                stream_name.into()
            };

            match access_kind {
                StreamAccessKind::Sync => target_name,
                StreamAccessKind::DiscreteWindow(w) => {
                    let window = mir.discrete_window(*w);
                    let target_name = mir.stream(window.target).name();
                    let duration = window.duration;
                    let op = &window.op;
                    format!("{target_name}.aggregate(over_discrete: {duration}, using: {op})")
                },
                StreamAccessKind::SlidingWindow(w) => {
                    let window = mir.sliding_window(*w);
                    let target_name = mir.stream(window.target).name();
                    let duration = window.duration.as_secs_f64().to_string();
                    let op = &window.op;
                    format!("{target_name}.aggregate(over: {duration}s, using: {op})")
                },
                StreamAccessKind::InstanceAggregation(w) => {
                    let window = mir.instance_aggregation(*w);
                    let target_name = mir.stream(window.target).name();
                    let duration = window.selection.to_string();
                    let op = &window.op().to_string();
                    format!("{target_name}.aggregate(over_instances: {duration}, using: {op})")
                },
                StreamAccessKind::Hold => format!("{target_name}.hold()"),
                StreamAccessKind::Offset(o) => format!("{target_name}.offset(by:-{o})"),
                StreamAccessKind::Get => format!("{target_name}.get()"),
                StreamAccessKind::Fresh => format!("{target_name}.fresh()"),
            }
        },
        ExpressionKind::ParameterAccess(sref, parameter) => mir.output(*sref).params[*parameter].name.to_string(),
        ExpressionKind::Ite {
            condition,
            consequence,
            alternative,
        } => {
            let display_condition = display_expression(mir, condition, 0);
            let display_consequence = display_expression(mir, consequence, 0);
            let display_alternative = display_expression(mir, alternative, 0);
            format!("if {display_condition} then {display_consequence} else {display_alternative}")
        },
        ExpressionKind::Tuple(exprs) => {
            let display_exprs = exprs
                .iter()
                .map(|expr| display_expression(mir, expr, 0))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({display_exprs})")
        },
        ExpressionKind::TupleAccess(expr, i) => {
            let display_expr = display_expression(mir, expr, 20);
            format!("{display_expr}({i})")
        },
        ExpressionKind::Function(name, args) => {
            let display_args = args
                .iter()
                .map(|arg| display_expression(mir, arg, 0))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{name}({display_args})")
        },
        ExpressionKind::Convert { expr: inner_expr } => {
            let inner_display = display_expression(mir, inner_expr, 0);
            format!("Cast<{},{}>({inner_display})", expr.ty, inner_expr.ty)
        },
        ExpressionKind::Default { expr, default } => {
            let display_expr = display_expression(mir, expr, 0);
            let display_default = display_expression(mir, default, 0);
            format!("{display_expr}.defaults(to: {display_default})")
        },
    }
}

impl<'a> Display for RtLolaMirPrinter<'a, Expression> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", display_expression(self.mir, self.inner, 0))
    }
}

impl Display for InputStream {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let name = &self.name;
        let ty = &self.ty;
        write!(f, "input {name} : {ty}")
    }
}

impl<'a> Display for RtLolaMirPrinter<'a, OutputStream> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let OutputStream {
            name,
            ty,
            spawn,
            eval,
            close,
            params,
            kind,
            ..
        } = self.inner;

        let display_parameters = if !params.is_empty() {
            let parameter_list = params
                .iter()
                .map(|parameter| format!("{} : {}", parameter.name, parameter.ty))
                .join(", ");
            format!("({parameter_list})")
        } else {
            "".into()
        };

        match kind {
            OutputKind::NamedOutput => write!(f, "output {name}{display_parameters} : {ty}")?,
            OutputKind::Trigger => write!(f, "trigger{display_parameters}")?,
        }

        if spawn.expression.is_some() || spawn.condition.is_some() {
            write!(f, "\n  spawn")?;
            if let Some(spawn_expr) = &spawn.expression {
                let display_spawn_expr = display_expression(self.mir, spawn_expr, 0);
                write!(f, " with {display_spawn_expr}")?;
            }
            if let Some(spawn_condition) = &spawn.condition {
                let display_spawn_condition = display_expression(self.mir, spawn_condition, 0);
                write!(f, " when {display_spawn_condition}")?;
            }
        }

        for clause in &eval.clauses {
            let display_pacing = RtLolaMirPrinter::new(self.mir, &eval.eval_pacing).to_string();
            write!(f, "\n  eval @{display_pacing} ")?;
            if let Some(eval_condition) = &clause.condition {
                let display_eval_condition = display_expression(self.mir, eval_condition, 0);
                write!(f, "when {display_eval_condition} ")?;
            }
            let display_eval_expr = display_expression(self.mir, &clause.expression, 0);
            write!(f, "with {display_eval_expr}")?;
        }

        if let Some(close_condition) = &close.condition {
            let display_close_condition = display_expression(self.mir, close_condition, 0);
            write!(f, "\n  close when {display_close_condition}")?;
        }

        Ok(())
    }
}

impl<'a> Display for RtLolaMirPrinter<'a, Trigger> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let output = self.mir.output(self.inner.output_reference);
        RtLolaMirPrinter::new(self.mir, output).fmt(f)
    }
}

impl Display for Mir {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.inputs.iter().try_for_each(|input| {
            RtLolaMirPrinter::new(self, input).fmt(f)?;
            write!(f, "\n\n")
        })?;

        self.outputs.iter().try_for_each(|output| {
            RtLolaMirPrinter::new(self, output).fmt(f)?;
            write!(f, "\n\n")
        })?;

        self.triggers.iter().try_for_each(|trigger| {
            RtLolaMirPrinter::new(self, trigger).fmt(f)?;
            write!(f, "\n\n")
        })
    }
}

#[cfg(test)]
mod tests {
    use rtlola_parser::ParserConfig;

    use super::display_expression;
    use crate::parse;

    macro_rules! test_display_expression {
        ( $( $name:ident: $test:expr => $expected:literal, )+) => {
            $(
            #[test]
            fn $name() {
                let spec = format!("input a : UInt64\noutput b@a := {}", $test);
                let config = ParserConfig::for_string(spec);
                let mir = parse(&config).expect("should parse");
                let expr = &mir.outputs[0].eval.clauses.get(0).expect("only one clause").expression;
                let display_expr = display_expression(&mir, expr, 0);
                assert_eq!(display_expr, $expected);
            }
            )+
        }
    }

    test_display_expression! {
        constant:
        "1" => "1",
        add:
        "1+2" => "1 + 2",
        mul:
        "1*2" => "1 * 2",
        add_after_mul:
        "1+2*3" => "1 + 2 * 3",
        mul_after_add:
        "1*(2+3)" => "1 * (2 + 3)",
        comparison1:
        "(1 > 2) && !false || (2 != 2)" => "1 > 2 ∧ !false ∨ 2 ≠ 2",
        comparison2:
        "(true == (1 > 2 || false)" => "true = (1 > 2 ∨ false)",
        associativity:
        "1 - (2 - 3)" => "1 - (2 - 3)",
        associativity2:
        "1 + (2 + 3)" => "1 + 2 + 3",
        sync_access:
        "a + 5" => "a + 5",
        hold_access:
        "a.hold().defaults(to: 0)" => "a.hold().defaults(to: 0)",
        offset_access:
        "a.offset(by:-2).defaults(to: 2+2)" => "a.offset(by:-2).defaults(to: 2 + 2)",
        floats:
        "1.0 + 1.5" => "1.0 + 1.5",
    }

    #[test]
    fn test() {
        let example = "input a : UInt64
        input b : UInt64
        output c@(a&&b) := a + b.hold().defaults(to:0)
        output d@10Hz := b.aggregate(over: 2s, using: sum) + c.hold().defaults(to:0)
        output e(x)
            spawn with a when b == 0
            eval when x == a with e(x).offset(by:-1).defaults(to:0) + 1
            close when x == a && e(x) == 0
        output f
            eval @a when a == 0 with 0
            eval @a when a > 5 && a <= 10 with 1
            eval @a when a > 10 with 2
        trigger c > 5 \"message\"
        ";

        let config = ParserConfig::for_string(example.into());
        let mir = parse(&config).expect("should parse");
        let config = ParserConfig::for_string(mir.to_string());
        parse(&config).expect("should also parse");
    }
}
