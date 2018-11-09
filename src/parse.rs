//! This module contains the parser for the Lola Language.

use ast::*;
use pest;
use pest::iterators::{Pair, Pairs};
use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;

#[derive(Parser)]
#[grammar = "lola.pest"]
pub(crate) struct LolaParser;

lazy_static! {
    // precedence taken from C/C++: https://en.wikipedia.org/wiki/Operators_in_C_and_C++
    // Precedence climber can be used to build the AST, see https://pest-parser.github.io/book/ for more details
    static ref PREC_CLIMBER: PrecClimber<Rule> = {
        use self::Assoc::*;
        use self::Rule::*;

        PrecClimber::new(vec![
            Operator::new(Or, Left),
            Operator::new(And, Left),
            Operator::new(Equal, Left) | Operator::new(NotEqual, Left),
            Operator::new(LessThan, Left) | Operator::new(LessThanOrEqual, Left) | Operator::new(MoreThan, Left) | Operator::new(MoreThanOrEqual, Left),
            Operator::new(Add, Left) | Operator::new(Subtract, Left),
            Operator::new(Multiply, Left) | Operator::new(Divide, Left) | Operator::new(Mod, Left),
            Operator::new(Power, Right),
        ])
    };
}

/**
 * Transforms a textual representation of a Lola specification into
 * an AST representation.
 */
pub(crate) fn parse(content: &str) -> Result<LolaSpec, pest::error::Error<Rule>> {
    let mut pairs = LolaParser::parse(Rule::Spec, content)?;
    let mut spec = LolaSpec::new();
    assert!(pairs.clone().count() == 1, "Spec must not be empty.");
    let spec_pair = pairs.next().unwrap();
    assert!(spec_pair.as_rule() == Rule::Spec);
    for pair in spec_pair.into_inner() {
        match pair.as_rule() {
            Rule::LanguageSpec => {
                spec.language = Some(LanguageSpec::from(pair.as_str()));
            }
            Rule::ConstantStream => {
                let constant = parse_constant(&mut spec, pair);
                spec.constants.push(constant);
            }
            Rule::InputStream => {
                let input = parse_inputs(&mut spec, pair);
                spec.inputs.extend(input);
            }
            Rule::OutputStream => {
                let output = parse_output(&mut spec, pair);
                spec.outputs.push(output);
            }
            Rule::Trigger => {
                let trigger = parse_trigger(&mut spec, pair);
                spec.trigger.push(trigger);
            }
            Rule::TypeDecl => {
                let type_decl = parse_type_declaration(&mut spec, pair);
                spec.type_declarations.push(type_decl);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(spec)
}

/**
 * Transforms a `Rule::ConstantStrean` into `Constant` AST node.
 * Panics if input is not `Rule::ConstantStrean`.
 * The constant rule consists of the following tokens:
 * - Rule::Ident
 * - Rule::Type
 * - Rule::Literal
 */
fn parse_constant(spec: &mut LolaSpec, pair: Pair<Rule>) -> Constant {
    assert_eq!(pair.as_rule(), Rule::ConstantStream);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
    let ty = parse_type(
        spec,
        pairs.next().expect("mismatch between grammar and AST"),
    );
    let literal = parse_literal(
        spec,
        pairs.next().expect("mismatch between grammar and AST"),
    );
    Constant {
        id: NodeId::DUMMY,
        name,
        ty: Some(ty),
        literal,
        span,
    }
}

/**
 * Transforms a `Rule::InputStrean` into `Input` AST node.
 * Panics if input is not `Rule::InputStrean`.
 * The input rule consists of non-empty sequences of following tokens:
 * - Rule::Ident
 * - (Rule::ParamList)?
 * - Rule::Type
 */
fn parse_inputs(spec: &mut LolaSpec, pair: Pair<Rule>) -> Vec<Input> {
    assert_eq!(pair.as_rule(), Rule::InputStream);
    let mut inputs = Vec::new();
    let mut pairs = pair.into_inner();
    while let Some(pair) = pairs.next() {
        let start = pair.as_span().start();
        let name = parse_ident(&pair);

        let pair = pairs.next().expect("mismatch between grammar and AST");
        let end = pair.as_span().end();
        let ty = parse_type(spec, pair);
        inputs.push(Input {
            id: NodeId::DUMMY,
            name,
            ty,
            span: Span { start, end },
        })
    }

    assert!(!inputs.is_empty());
    inputs
}

/**
 * Transforms a `Rule::OutputStream` into `Output` AST node.
 * Panics if input is not `Rule::OutputStream`.
 * The output rule consists of the following tokens:
 * - Rule::Ident
 * - Rule::Type
 * - Rule::Expr
 */
fn parse_output(spec: &mut LolaSpec, pair: Pair<Rule>) -> Output {
    assert_eq!(pair.as_rule(), Rule::OutputStream);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(&pairs.next().expect("mismatch between grammar and AST"));

    let mut pair = pairs.next().expect("mismatch between grammar and AST");
    let params = if let Rule::ParamList = pair.as_rule() {
        let res = parse_parameter_list(spec, pair.into_inner());
        pair = pairs.next().expect("mismatch between grammar and AST");
        res
    } else {
        Vec::new()
    };

    let mut ty = None;
    if let Rule::Type = pair.as_rule() {
        ty = Some(parse_type(spec, pair));
        pair = pairs.next().expect("mismatch between grammar and AST");
    }

    let mut tspec = None;
    if let Rule::TemplateSpec = pair.as_rule() {
        tspec = Some(parse_template_spec(spec, pair));
        pair = pairs.next().expect("mismatch between grammar and AST");
    };

    // Parse expression
    let expr_span = pair.as_span();
    let expression = build_expression_ast(spec, pair.into_inner(), expr_span.into());
    Output {
        id: NodeId::DUMMY,
        name,
        ty,
        params,
        template_spec: tspec,
        expression,
        span,
    }
}

fn parse_parameter_list(spec: &mut LolaSpec, param_list: Pairs<Rule>) -> Vec<Parameter> {
    let mut params = Vec::new();
    for param_decl in param_list {
        assert_eq!(Rule::ParameterDecl, param_decl.as_rule());
        let mut decl = param_decl.into_inner();
        let name = parse_ident(&decl.next().expect("mismatch between grammar and AST"));
        let ty = if let Some(type_pair) = decl.next() {
            assert_eq!(Rule::Type, type_pair.as_rule());
            Some(parse_type(spec, type_pair))
        } else {
            None
        };
        params.push(Parameter { name, ty });
    }
    params
}

fn parse_template_spec(spec: &mut LolaSpec, pair: Pair<Rule>) -> TemplateSpec {
    let mut decls = pair.into_inner();
    let mut pair = decls.next();
    let mut rule = pair.as_ref().map(|p| p.as_rule());

    let mut inv_spec = None;
    if let Some(Rule::InvokeDecl) = rule {
        inv_spec = Some(parse_inv_spec(spec, pair.unwrap()));
        pair = decls.next();
        rule = pair.as_ref().map(|p| p.as_rule());
    }
    let mut ext_spec = None;
    if let Some(Rule::ExtendDecl) = rule {
        ext_spec = Some(parse_ext_spec(spec, pair.unwrap()));
        pair = decls.next();
        rule = pair.as_ref().map(|p| p.as_rule());
    }
    let mut ter_spec = None;
    if let Some(Rule::TerminateDecl) = rule {
        let expr = pair
            .unwrap()
            .into_inner()
            .next()
            .expect("mismatch between grammar and AST");
        let expr_span = expr.as_span().into();
        let expr = build_expression_ast(spec, expr.into_inner(), expr_span);
        ter_spec = Some(TerminateSpec { target: expr });
    }
    TemplateSpec {
        inv: inv_spec,
        ext: ext_spec,
        ter: ter_spec,
    }
}

fn parse_frequency(spec: &mut LolaSpec, freq: Pair<Rule>) -> ExtendRate {
    let freq_rule = freq.as_rule();
    let mut children = freq.into_inner();
    let expr = children.next().expect("mismatch between grammar and AST");
    let span = expr.as_span().into();
    let expr = build_expression_ast(spec, expr.into_inner(), span);
    let unit_pair = children.next().expect("mismatch between grammar and AST");
    let unit_str = unit_pair.as_str();
    match freq_rule {
        Rule::Frequency => {
            assert_eq!(unit_pair.as_rule(), Rule::UnitOfFreq);
            let unit = parse_frequency_unit(unit_str);
            ExtendRate::Frequency(Box::new(expr), unit)
        }
        Rule::Duration => {
            assert_eq!(unit_pair.as_rule(), Rule::UnitOfTime);
            let unit = parse_duration_unit(unit_str);
            ExtendRate::Duration(Box::new(expr), unit)
        }
        _ => unreachable!(),
    }
}

fn parse_ext_spec(spec: &mut LolaSpec, ext_pair: Pair<Rule>) -> ExtendSpec {
    let mut children = ext_pair.into_inner();
    let first_child = children.next().expect("mismatch between grammar and ast");

    let mut freq = None;
    let mut target = None;
    match first_child.as_rule() {
        Rule::Frequency | Rule::Duration => freq = Some(parse_frequency(spec, first_child)),
        Rule::Expr => {
            let span = first_child.as_span().into();
            target = Some(build_expression_ast(spec, first_child.into_inner(), span));
            if let Some(freq_pair) = children.next() {
                freq = Some(parse_frequency(spec, freq_pair));
            }
        }
        _ => unreachable!(),
    }

    assert!(freq.is_some() || target.is_some());
    ExtendSpec { target, freq }
}

fn parse_inv_spec(spec: &mut LolaSpec, inv_pair: Pair<Rule>) -> InvokeSpec {
    let mut inv_children = inv_pair.into_inner();
    let expr_pair = inv_children
        .next()
        .expect("mismatch between grammar and AST");
    let expr_span = expr_pair.as_span().into();
    let inv_target = build_expression_ast(spec, expr_pair.into_inner(), expr_span);
    // Compute invocation condition:
    let mut is_if = false;
    let mut cond_expr = None;
    if let Some(inv_cond_pair) = inv_children.next() {
        is_if = match inv_cond_pair.as_rule() {
            Rule::InvokeIf => true,
            Rule::InvokeUnless => false,
            _ => unreachable!(),
        };
        let condition = inv_cond_pair
            .into_inner()
            .next()
            .expect("mismatch between grammar and AST");
        let cond_expr_span = condition.as_span().into();
        cond_expr = Some(build_expression_ast(
            spec,
            condition.into_inner(),
            cond_expr_span,
        ))
    }
    InvokeSpec {
        condition: cond_expr,
        is_if,
        target: inv_target,
    }
}

/**
 * Transforms a `Rule::Trigger` into `Trigger` AST node.
 * Panics if input is not `Rule::Trigger`.
 * The output rule consists of the following tokens:
 * - (Rule::Ident)?
 * - Rule::Expr
 * - (Rule::StringLiteral)?
 */
fn parse_trigger(spec: &mut LolaSpec, pair: Pair<Rule>) -> Trigger {
    assert_eq!(pair.as_rule(), Rule::Trigger);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();

    let mut name = None;
    let mut message = None;

    let mut pair = pairs.next().expect("mismatch between grammar and AST");
    // first token is either expression or identifier
    if let Rule::Ident = pair.as_rule() {
        name = Some(parse_ident(&pair));
        pair = pairs.next().expect("mismatch between grammar and AST");
    }
    let expr_span = pair.as_span();
    let expression = build_expression_ast(spec, pair.into_inner(), expr_span.into());

    if let Some(pair) = pairs.next() {
        assert_eq!(pair.as_rule(), Rule::String);
        message = Some(pair.as_str().to_string());
    }

    Trigger {
        id: NodeId::DUMMY,
        name,
        expression,
        message,
        span,
    }
}

/**
 * Transforms a `Rule::Ident` into `Ident` AST node.
 * Panics if input is not `Rule::Ident`.
 */
fn parse_ident(pair: &Pair<Rule>) -> Ident {
    assert_eq!(pair.as_rule(), Rule::Ident);
    let name = pair.as_str().to_string();
    Ident::new(name, pair.as_span().into())
}

/**
 * Transforms a `Rule::TypeDecl` into `TypeDeclaration` AST node.
 * Panics if input is not `Rule::TypeDecl`.
 */
fn parse_type_declaration(spec: &mut LolaSpec, pair: Pair<Rule>) -> TypeDeclaration {
    assert_eq!(pair.as_rule(), Rule::TypeDecl);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
    let mut fields = Vec::new();
    while let Some(pair) = pairs.next() {
        let field_name = pair.as_str().to_string();
        let ty = parse_type(
            spec,
            pairs.next().expect("mismatch between grammar and AST"),
        );
        fields.push(Box::new(TypeDeclField {
            name: field_name,
            ty,
        }));
    }
    let kind = TypeKind::UserDefined(fields);
    TypeDeclaration {
        name: Some(name),
        kind,
        span,
    }
}

/**
 * Transforms a `Rule::Type` into `Type` AST node.
 * Panics if input is not `Rule::Type`.
 */
fn parse_type(spec: &mut LolaSpec, pair: Pair<Rule>) -> Type {
    assert_eq!(pair.as_rule(), Rule::Type);
    let span = pair.as_span();
    let mut tuple = Vec::new();
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::Ident => {
                let ty = Type::new_simple(pair.as_str().to_string(), pair.as_span().into());
                return ty;
            }
            Rule::Type => tuple.push(Box::new(parse_type(spec, pair))),
            _ => unreachable!(),
        }
    }
    Type::new_tuple(tuple, span.into())
}

/**
 * Transforms a `Rule::Literal` into `Literal` AST node.
 * Panics if input is not `Rule::Literal`.
 */
fn parse_literal(spec: &mut LolaSpec, pair: Pair<Rule>) -> Literal {
    assert_eq!(pair.as_rule(), Rule::Literal);
    let inner = pair
        .into_inner()
        .next()
        .expect("Rule::Literal has exactly one child");
    match inner.as_rule() {
        Rule::String => unimplemented!(),
        Rule::NumberLiteral => {
            let str_rep = inner.as_str();
            if let Result::Ok(i) = str_rep.parse::<i128>() {
                return Literal::new_int(i, inner.as_span().into());
            } else if let Result::Ok(f) = str_rep.parse::<f64>() {
                return Literal::new_float(f, inner.as_span().into());
            } else {
                panic!("Number literal not valid in rust.")
            }
        }
        Rule::TupleLiteral => {
            let span = inner.as_span();
            let elements = inner.into_inner();
            let literals: Vec<Literal> = elements.map(|pair| parse_literal(spec, pair)).collect();
            Literal::new_tuple(&literals, span.into())
        }
        Rule::True => Literal::new_bool(true, inner.as_span().into()),
        Rule::False => Literal::new_bool(false, inner.as_span().into()),
        _ => unreachable!(),
    }
}

fn parse_stream_instance(spec: &mut LolaSpec, instance: Pair<Rule>) -> StreamInstance {
    let mut children = instance.into_inner();
    // Parse the stream identifier in isolation.
    let stream_ident = parse_ident(&children.next().unwrap());
    // Parse remaining children, aka the arguments.
    let args = parse_vec_of_expressions(spec, children);
    StreamInstance {
        stream_identifier: stream_ident,
        arguments: args,
    }
}

fn parse_vec_of_expressions(spec: &mut LolaSpec, pairs: Pairs<Rule>) -> Vec<Box<Expression>> {
    pairs
        .map(|expr| {
            let span = expr.as_span().into();
            build_expression_ast(spec, expr.into_inner(), span)
        }).map(Box::new)
        .collect()
}

fn parse_duration_unit(str: &str) -> TimeUnit {
    match str {
        "ns" => TimeUnit::NanoSecond,
        "μs" => TimeUnit::MicroSecond,
        "ms" => TimeUnit::MilliSecond,
        "s" => TimeUnit::Second,
        "min" => TimeUnit::Minute,
        "h" => TimeUnit::Hour,
        "d" => TimeUnit::Day,
        "w" => TimeUnit::Week,
        "a" => TimeUnit::Year,
        _ => unreachable!(),
    }
}

fn parse_frequency_unit(str: &str) -> FreqUnit {
    match str {
        "μHz" => FreqUnit::MicroHertz,
        "mHz" => FreqUnit::MilliHertz,
        "Hz" => FreqUnit::Hertz,
        "kHz" => FreqUnit::KiloHertz,
        "MHz" => FreqUnit::MegaHertz,
        "GHz" => FreqUnit::GigaHertz,
        _ => unreachable!(),
    }
}

fn parse_lookup_expression(spec: &mut LolaSpec, pair: Pair<Rule>, span: Span) -> Expression {
    let mut children = pair.into_inner();
    let stream_instance = children
        .next()
        .expect("Lookups need to have a target stream instance.");
    let stream_instance = parse_stream_instance(spec, stream_instance);
    let second_child = children.next().unwrap();
    let second_child_span = second_child.as_span();
    match second_child.as_rule() {
        Rule::Expr => {
            // Discrete offset
            let offset =
                build_expression_ast(spec, second_child.into_inner(), second_child_span.into());
            let offset = Offset::DiscreteOffset(Box::new(offset));
            Expression::new(ExpressionKind::Lookup(stream_instance, offset, None), span)
        }
        Rule::Duration => {
            // Real time offset
            let mut duration_children = second_child.into_inner();
            let time_interval = duration_children
                .next()
                .expect("Duration needs a time span.");
            let time_interval_span = time_interval.as_span().into();
            let time_interval =
                build_expression_ast(spec, time_interval.into_inner(), time_interval_span);
            let unit_string = duration_children
                .next()
                .expect("Duration needs a time unit.")
                .as_str();
            let unit = parse_duration_unit(unit_string);
            let offset = Offset::RealTimeOffset(Box::new(time_interval), unit);
            // Now check whether it is a window or not.
            let aggregation = match children.next().map(|x| x.as_rule()) {
                Some(Rule::Sum) => Some(WindowOperation::Sum),
                Some(Rule::Product) => Some(WindowOperation::Product),
                Some(Rule::Average) => Some(WindowOperation::Average),
                Some(Rule::Count) => Some(WindowOperation::Count),
                Some(Rule::Integral) => Some(WindowOperation::Integral),
                None => None,
                _ => unreachable!(),
            };
            Expression::new(
                ExpressionKind::Lookup(stream_instance, offset, aggregation),
                span,
            )
        }
        _ => unreachable!(),
    }
}

fn build_function_expression(spec: &mut LolaSpec, pair: Pair<Rule>, span: Span) -> Expression {
    let mut children = pair.into_inner();
    let name = children.next().unwrap().as_str();
    let function_kind = match name {
        "nroot" => FunctionKind::NthRoot,
        "sqrt" => FunctionKind::Sqrt,
        "π" => FunctionKind::Projection,
        "sin" => FunctionKind::Sin,
        "cos" => FunctionKind::Cos,
        "tan" => FunctionKind::Tan,
        "arcsin" => FunctionKind::Arcsin,
        "arccos" => FunctionKind::Arccos,
        "arctar" => FunctionKind::Arctan,
        "exp" => FunctionKind::Exp,
        "floor" => FunctionKind::Floor,
        "ceil" => FunctionKind::Ceil,
        _ => panic!("Unknown function symbol: {}.", name),
    };
    let args = parse_vec_of_expressions(spec, children);
    Expression::new(ExpressionKind::Function(function_kind, args), span)
}

/**
 * Builds the Expr AST.
 */
fn build_expression_ast(spec: &mut LolaSpec, pairs: Pairs<Rule>, span: Span) -> Expression {
    PREC_CLIMBER.climb(
        pairs,
        |pair: Pair<Rule>| {
            let span = pair.as_span();
            match pair.as_rule() { // Map function from `Pair` to AST data structure `Expression`
                Rule::Literal => {
                    Expression::new(ExpressionKind::Lit(parse_literal(spec, pair)), span.into())
                }
                Rule::Ident => {
                    Expression::new(ExpressionKind::Ident(parse_ident(&pair)), span.into())
                }
                Rule::ParenthesizedExpression => {
                    let mut inner = pair
                        .into_inner();
                    let opp = inner.next().expect("Rule::ParenthesizedExpression has a token for the (potentialy missing) opening parenthesis");
                    let opening_parenthesis  = if let Rule::OpeningParenthesis = opp.as_rule() {
                        Some(Box::new(Parenthesis::new(opp.as_span().into())))
                    } else{
                        None
                    };

                    let inner_expression = inner.next().expect("Rule::ParenthesizedExpression has a token for the contained expression");

                    let closing = inner.next().expect("Rule::ParenthesizedExpression has a token for the (potentialy missing) closing parenthesis");
                    let closing_parenthesis  = if let Rule::ClosingParenthesis = closing.as_rule() {
                    Some(Box::new(Parenthesis::new(closing.as_span().into())))
                    }
                    else{
                        None
                    };

                    let inner_span = inner_expression.as_span().into();
                    Expression::new(
                        ExpressionKind::ParenthesizedExpression(
                            opening_parenthesis,
                            Box::new(build_expression_ast(spec, inner_expression.into_inner(), inner_span)),
                            closing_parenthesis
                        ),
                        span.into())
                },
                Rule::DefaultExpr => {
                    let mut children = pair.into_inner();
                    let lookup = children.next().unwrap();
                    let lookup_span = lookup.as_span().into();
                    let default = children.next().unwrap();
                    let default_span = default.as_span().into();
                    let lookup = parse_lookup_expression(spec, lookup, lookup_span);
                    let default = build_expression_ast(spec, default.into_inner(), default_span);
                    Expression::new(ExpressionKind::Default(Box::new(lookup), Box::new(default)), span.into())
                },
                Rule::LookupExpr => parse_lookup_expression(spec, pair, span.into()),
                Rule::UnaryExpr => { // First child is the operator, second the operand.
                    let mut children = pair.into_inner();
                    let pest_operator = children.next().expect("Unary expressions need to have an operator.");
                    let operand = children.next().expect("Unary expressions need to have an operand.");
                    let op_span = operand.as_span().into();
                    let operand = build_expression_ast(spec, operand.into_inner(), op_span);
                    let operator = match pest_operator.as_rule() {
                        Rule::Add => return operand, // Discard unary plus because it is semantically null.
                        Rule::Subtract => UnOp::Neg,
                        Rule::Neg => UnOp::Not,
                        _ => unreachable!(),
                    };
                    Expression::new(ExpressionKind::Unary(operator, Box::new(operand)), span.into())
                },
                Rule::TernaryExpr => {
                    let mut children = parse_vec_of_expressions(spec, pair.into_inner());
                    assert_eq!(children.len(), 3, "A ternary expression needs exactly three children.");
                    Expression::new(ExpressionKind::Ite(children.remove(0), children.remove(0), children.remove(0)), span.into())
                },
                Rule::Tuple => {
                    let elements = parse_vec_of_expressions(spec, pair.into_inner());
                    assert!(elements.len() != 1, "Tuples may not have exactly one element.");
                    Expression::new(ExpressionKind::Tuple(elements), span.into())
                },
                Rule::Expr => {
                    let span = pair.as_span();
                    build_expression_ast(spec, pair.into_inner(), span.into())
                }
                Rule::FunctionExpr => build_function_expression(spec, pair, span.into()),
                _ => panic!("Unexpected rule when parsing expression ast: {:?}", pair.as_rule()),
            }
        },
        |lhs: Expression, op: Pair<Rule>, rhs: Expression| { // Reduce function combining `Expression`s to `Expression`s with the correct precs
            let op = match op.as_rule() {
                Rule::Add => BinOp::Add,
                Rule::Subtract => BinOp::Sub,
                Rule::Multiply => BinOp::Mul,
                Rule::Divide => BinOp::Div,
                Rule::Mod => BinOp::Rem,
                Rule::Power => BinOp::Pow,
                Rule::And => BinOp::And,
                Rule::Or => BinOp::Or,
                Rule::LessThan => BinOp::Lt,
                Rule::LessThanOrEqual => BinOp::Le,
                Rule::MoreThan => BinOp::Gt,
                Rule::MoreThanOrEqual => BinOp::Ge,
                Rule::Equal => BinOp::Eq,
                Rule::NotEqual => BinOp::Ne,
                _ => unreachable!(),
            };
            Expression::new(
                ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                span,
            )
        },
    )
}

#[derive(Debug)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: String, span: Span) -> Ident {
        Ident { name, span }
    }
}

/// A span marks a range in a file.
/// Start and end positions are *byte* offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    // TODO Do we need this here or do we want to keep a mapping from byte positions to lines in the LSP part.
    // line: usize,
    // /// The LSP uses UTF-16 code units (2 bytes) as their unit for offsets.
    // lineOffsetLSP: usize,
}

impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span {
            start: span.start(),
            end: span.end(),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn cmp_ast_spec(ast: LolaSpec, spec: &str) -> bool {
        // Todo: Make more robust, e.g. against changes in whitespace.
        assert_eq!(format!("{}", ast), spec.replace("\n", " "));
        true
    }

    #[test]
    fn parse_simple() {
        let _ = LolaParser::parse(
            Rule::Spec,
            "input in: Int\noutput out: Int := in\ntrigger in != out",
        ).unwrap_or_else(|e| panic!("{}", e));
    }

    #[test]
    fn parse_constant() {
        parses_to! {
            parser: LolaParser,
            input:  "constant five : Int := 5",
            rule:   Rule::ConstantStream,
            tokens: [
                ConstantStream(0, 24, [
                    Ident(9, 13, []),
                    Type(16, 19, [
                        Ident(16, 19, []),
                    ]),
                    Literal(23, 24, [
                        NumberLiteral(23, 24, []),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_constant_ast() {
        let pair = LolaParser::parse(Rule::ConstantStream, "constant five : Int := 5")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_constant(&mut spec, pair);
        assert_eq!(format!("{}", ast), "constant five: Int := 5")
    }

    #[test]
    fn parse_constant_double() {
        let pair = LolaParser::parse(Rule::ConstantStream, "constant fiveoh: Double := 5.0")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_constant(&mut spec, pair);
        assert_eq!(format!("{}", ast), "constant fiveoh: Double := 5.0")
    }

    #[test]
    fn parse_input() {
        parses_to! {
            parser: LolaParser,
            input:  "input in: Int",
            rule:   Rule::InputStream,
            tokens: [
                InputStream(0, 13, [
                    Ident(6, 8, []),
                    Type(10, 13, [
                        Ident(10, 13, []),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_input_ast() {
        let pair = LolaParser::parse(Rule::InputStream, "input a: Int, b: Int, c: Bool")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let inputs = super::parse_inputs(&mut spec, pair);
        assert_eq!(inputs.len(), 3);
        assert_eq!(format!("{}", inputs[0]), "input a: Int");
        assert_eq!(format!("{}", inputs[1]), "input b: Int");
        assert_eq!(format!("{}", inputs[2]), "input c: Bool");
    }

    #[test]
    fn parse_output() {
        parses_to! {
            parser: LolaParser,
            input:  "output out: Int := in + 1",
            rule:   Rule::OutputStream,
            tokens: [
                OutputStream(0, 25, [
                    Ident(7, 10, []),
                    Type(12, 15, [
                        Ident(12, 15, []),
                    ]),
                    Expr(19, 25, [
                        Ident(19, 21, []),
                        Add(22, 23, []),
                        Literal(24, 25, [
                            NumberLiteral(24, 25, []),
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_output_ast() {
        let pair = LolaParser::parse(Rule::OutputStream, "output out: Int := in + 1")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_output(&mut spec, pair);
        assert_eq!(format!("{}", ast), "output out: Int := in + 1")
    }

    #[test]
    fn parse_trigger() {
        parses_to! {
            parser: LolaParser,
            input:  "trigger in != out \"some message\"",
            rule:   Rule::Trigger,
            tokens: [
                Trigger(0, 32, [
                    Expr(8, 17, [
                        Ident(8, 10, []),
                        NotEqual(11, 13, []),
                        Ident(14, 17, []),
                    ]),
                    String(19, 31, []),
                ]),
            ]
        };
    }

    #[test]
    fn parse_trigger_ast() {
        let pair = LolaParser::parse(Rule::Trigger, "trigger in != out \"some message\"")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_trigger(&mut spec, pair);
        assert_eq!(format!("{}", ast), "trigger in != out \"some message\"")
    }

    #[test]
    fn parse_expression() {
        let expr = LolaParser::parse(Rule::Expr, "in + 1\n")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        assert_eq!(format!("{}", ast), "in + 1")
    }

    #[test]
    fn parse_expression_precedence() {
        let expr = LolaParser::parse(Rule::Expr, "(a ∨ b ∧ c)")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        assert_eq!(format!("{}", ast), "(a ∨ b ∧ c)")
    }

    #[test]
    fn parse_missing_closing_parenthesis() {
        let expr = LolaParser::parse(Rule::Expr, "(a ∨ b ∧ c")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        assert_eq!(format!("{}", ast), "(a ∨ b ∧ c")
    }

    #[test]
    fn build_simple_ast() {
        let spec = "input in: Int\noutput out: Int := in\ntrigger in != out";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_ast_input() {
        let spec = "input in: Int\ninput in2: Int\ninput in3: (Int, Bool)\ninput in4: Bool";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_parenthesized_expression() {
        let spec = "output s: Bool := (true ∨ true)";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_lookup_expression() {
        let spec = "output s: Int := s[-1] ? (3 * 4)";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_ternary_expression() {
        let spec = "input in: Int\noutput s: Int := if in = 3 then 4 else in + 2";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_function_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := nroot(1, π(1, in))";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_trigger() {
        let spec = "input in: Int\ntrigger in > 5";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_complex_expression() {
        let spec = "output s: Double := if !((s[-1] ? (3 * 4) + -4) = 12) ∨ true = false then 2.0 else 4.1";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_type_declaration() {
        let spec = "type VerifiedUser { name: String }";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_parameter_list() {
        let spec = "output s <a: B, c: D>: E := 3";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }

    #[test]
    fn build_template_spec() {
        let spec = "output s: Int { invoke inp unless 3 > 5 extend b @ 5GHz terminate false } := 3";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(ast, spec);
    }
}
