use std::fmt::{Display, Formatter, Result};

use super::{FloatTy, InputStream, IntTy, Mir, OutputStream, PacingType, Stream, Trigger, UIntTy, WindowOperation};
use crate::mir::{
    ActivationCondition, ArithLogOp, Constant, Expression, ExpressionKind, Offset, StreamAccessKind, Type,
};

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            ExpressionKind::LoadConstant(c) => write!(f, "{}", c),
            ExpressionKind::Function(name, args) => {
                write!(f, "{}(", name)?;
                if let Type::Function {
                    args: arg_tys,
                    ret: res,
                } = &self.ty
                {
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
            ExpressionKind::StreamAccess {
                target: sr,
                access_kind: access,
                parameters: para,
            } => {
                assert!(para.is_empty());
                match access {
                    StreamAccessKind::Sync => write!(f, "{}", sr),
                    StreamAccessKind::Hold => write!(f, "{}.hold()", sr),
                    StreamAccessKind::Get => write!(f, "{}.get()", sr),
                    StreamAccessKind::Fresh => write!(f, "{}.is_fresh()", sr),
                    StreamAccessKind::Offset(offset) => write!(f, "{}.offset({})", sr, offset),
                    StreamAccessKind::SlidingWindow(wr) | StreamAccessKind::DiscreteWindow(wr) => write!(f, "{}", wr),
                }
            },
            ExpressionKind::ParameterAccess(sr, idx) => {
                write!(f, "Parameter({}, {})", sr, idx)
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
            Type::Function { args, ret } => write_delim_list(f, args, "(", &format!(") -> {}", ret), ","),
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
/// Uses the formatter.
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

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Offset::Past(u) => write!(f, "{}", u),
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

pub(crate) fn display_ac(mir: &Mir, ac: &ActivationCondition) -> String {
    match ac {
        ActivationCondition::Conjunction(s) => s
            .iter()
            .map(|ac| display_ac(mir, ac))
            .collect::<Vec<_>>()
            .join(&ArithLogOp::And.to_string()),
        ActivationCondition::Disjunction(s) => s
            .iter()
            .map(|ac| display_ac(mir, ac))
            .collect::<Vec<_>>()
            .join(&ArithLogOp::Or.to_string()),
        ActivationCondition::Stream(s) => mir.stream(*s).name().into(),
        ActivationCondition::True => "true".into(),
    }
}

pub(crate) fn display_pacing_type(mir: &Mir, pacing_type: &PacingType) -> String {
    match pacing_type {
        super::PacingType::Periodic(f) => {
            let s = f
                .into_format_args(uom::si::frequency::hertz, uom::fmt::DisplayStyle::Abbreviation)
                .to_string();
            format!("{}Hz", &s[..s.len() - 3]) // TODO: better solution
        },
        super::PacingType::Event(ac) => display_ac(mir, ac),
        super::PacingType::Constant => "true".into(),
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

    let associativity = match op {
        ArithLogOp::Div | ArithLogOp::Sub => false,
        _ => true,
    };

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
                StreamAccessKind::Sync => format!("{target_name}"),
                StreamAccessKind::DiscreteWindow(_) => todo!(),
                StreamAccessKind::SlidingWindow(w) => {
                    let window = mir.sliding_window(*w);
                    let target_name = mir.stream(window.target).name();
                    let duration = window.duration.as_secs();
                    let op = &window.op;
                    format!("{target_name}.aggregate(over: {duration}s, using: {op})")
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

fn display_input(_: &Mir, input: &InputStream) -> String {
    let name = &input.name;
    let ty = &input.ty;
    format!("input {name} : {ty}")
}

fn display_output(mir: &Mir, output: &OutputStream) -> String {
    let name = &output.name;
    let ty = &output.ty;
    let pacing = &output.eval.eval_pacing;
    let spawn_expr = &output.spawn.expression;
    let spawn_condition = &output.spawn.condition;
    let eval_expr = &output.eval.expression;
    let eval_condition = &output.eval.condition;
    let close_condition = &output.close.condition;
    let parameters = &output.params;

    let display_pacing = display_pacing_type(mir, pacing);
    let display_parameters = if !parameters.is_empty() {
        let parameter_list = parameters
            .iter()
            .map(|parameter| format!("{} : {}", parameter.name, parameter.ty))
            .collect::<Vec<_>>()
            .join(", ");
        format!("({parameter_list})")
    } else {
        "".into()
    };

    let mut s = format!("output {name}{display_parameters} : {ty}\n");

    if let Some(spawn_expr) = spawn_expr {
        let display_spawn_expr = display_expression(mir, spawn_expr, 0);
        s.push_str(&format!("  spawn with {display_spawn_expr}"));
        if let Some(spawn_condition) = spawn_condition {
            let display_spawn_condition = display_expression(mir, spawn_condition, 0);
            s.push_str(&format!(" when {display_spawn_condition}"));
        }
        s.push('\n');
    }

    s.push_str(&format!("  eval @{display_pacing} "));
    if let Some(eval_condition) = eval_condition {
        let display_eval_condition = display_expression(mir, eval_condition, 0);
        s.push_str(&format!("when {display_eval_condition} "));
    }
    let display_eval_expr = display_expression(mir, eval_expr, 0);
    s.push_str(&format!("with {display_eval_expr}"));

    if let Some(close_condition) = close_condition {
        let display_close_condition = display_expression(mir, close_condition, 0);
        s.push_str(&format!("\n  close when {display_close_condition}"));
    }

    s
}

fn display_trigger(mir: &Mir, trigger: &Trigger) -> String {
    let output = mir.output(trigger.reference);
    let output_name = output.name();
    let message = &trigger.message;

    let s = format!("trigger {output_name} \"{message}\"");

    s
}

impl Display for Mir {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let inputs = self.inputs.iter().map(|input| display_input(self, input));
        let outputs = self.outputs.iter().map(|output| display_output(self, output));
        let triggers = self.triggers.iter().map(|trigger| display_trigger(self, trigger));

        let s = inputs
            .chain(outputs)
            .chain(triggers)
            .collect::<Vec<String>>()
            .join("\n\n");
        f.write_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::display_expression;
    use crate::parse;
    use rtlola_parser::ParserConfig;

    macro_rules! test_display_expression {
        ( $( $name:ident: $test:expr => $expected:literal, )+) => {
            $(
            #[test]
            fn $name() {
                let spec = format!("input a : UInt64\noutput b@a := {}", $test);
                let config = ParserConfig::for_string(spec);
                let mir = parse(config).expect("should parse");
                let expr = &mir.outputs[0].eval.expression;
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
    }

    #[test]
    fn test() {
        let example = "input a : UInt64
        input b : UInt64
        output c := a + b.hold().defaults(to:0)
        output d@10Hz := b.aggregate(over: 2s, using: sum) + c.hold().defaults(to:0)
        output e(x)
            spawn with a when b == 0
            eval when x == a with e(x).offset(by:-1).defaults(to:0) + 1
            close when x == a && e(x) == 0
        trigger c > 5 \"message\"
        ";

        let config = ParserConfig::for_string(example.into());
        let mir = parse(config).expect("should parse");
        println!("{mir}");
        let config = ParserConfig::for_string(mir.to_string());
        parse(config).expect("should also parse");
    }
}
