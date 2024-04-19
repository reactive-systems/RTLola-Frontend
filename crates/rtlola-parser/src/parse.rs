//! This module contains the parser for the Lola Language.

use std::convert::TryInto;
use std::rc::Rc;
use std::str::FromStr;

use lazy_static::lazy_static;
use num::rational::Rational64 as Rational;
use num::traits::Pow;
use num::{BigInt, BigRational, FromPrimitive, ToPrimitive};
use pest::iterators::{Pair, Pairs};
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;
use pest_derive::Parser;
use rtlola_reporting::{Diagnostic, RtLolaError, Span};

use super::ast::*;
use crate::ast::Literal;
use crate::syntactic_sugar::Desugarizer;
use crate::ParserConfig;

#[derive(Parser)]
#[grammar = "lola.pest"]
struct LolaParser;

#[derive(Debug, Clone)]
pub(crate) struct RtLolaParser<'a> {
    spec: RtLolaAst,
    config: &'a ParserConfig,
}

lazy_static! {
    static ref PRATT_PARSER: PrattParser<Rule> = {
        use self::Assoc::*;
        use self::Rule::*;

        PrattParser::new()
            .op(Op::infix(Implies, Right))
            .op(Op::infix(Or, Left))
            .op(Op::infix(And, Left))
            .op(Op::infix(BitOr, Left))
            .op(Op::infix(BitXor, Left))
            .op(Op::infix(BitAnd, Left))
            .op(Op::infix(Equal, Left) | Op::infix(NotEqual, Left))
            .op(Op::infix(LessThan, Left)
                | Op::infix(LessThanOrEqual, Left)
                | Op::infix(MoreThan, Left)
                | Op::infix(MoreThanOrEqual, Left))
            .op(Op::infix(ShiftLeft, Left) | Op::infix(ShiftRight, Left))
            .op(Op::infix(Add, Left) | Op::infix(Subtract, Left))
            .op(Op::infix(Multiply, Left) | Op::infix(Divide, Left) | Op::infix(Mod, Left))
            .op(Op::infix(Power, Right))
            .op(Op::infix(Dot, Left))
            .op(Op::infix(OpeningBracket, Left))
    };
}

impl<'a> RtLolaParser<'a> {
    pub(crate) fn new(config: &'a ParserConfig) -> Self {
        RtLolaParser {
            spec: RtLolaAst::empty(),
            config,
        }
    }

    /// Transforms a textual representation of a Lola specification into
    /// an AST representation.
    pub(crate) fn parse(config: &ParserConfig) -> Result<RtLolaAst, RtLolaError> {
        RtLolaParser::new(config)
            .parse_spec()
            .map(|ast| Desugarizer::all().remove_syn_sugar(ast))
    }

    /// Runs the parser on the give spec.
    pub(crate) fn parse_spec(mut self) -> Result<RtLolaAst, RtLolaError> {
        let mut pairs = LolaParser::parse(Rule::Spec, &self.config.spec).map_err(to_rtlola_error)?;
        assert!(pairs.clone().count() == 1, "Spec must not be empty.");
        let spec_pair = pairs.next().unwrap();
        assert!(spec_pair.as_rule() == Rule::Spec);
        let mut error = RtLolaError::new();
        for pair in spec_pair.into_inner() {
            match pair.as_rule() {
                Rule::ImportStmt => {
                    let import = self.parse_import(pair);
                    self.spec.imports.push(import);
                },
                Rule::ConstantStream => {
                    let constant = self.parse_constant(pair);
                    self.spec.constants.push(Rc::new(constant));
                },
                Rule::InputStream => {
                    let inputs = self.parse_inputs(pair);
                    self.spec.inputs.extend(inputs.into_iter().map(Rc::new));
                },
                Rule::MirrorStream => {
                    match self.parse_mirror(pair) {
                        Ok(mirror) => self.spec.mirrors.push(Rc::new(mirror)),
                        Err(e) => error.join(e),
                    }
                },
                Rule::OutputStream => {
                    match self.parse_output(pair) {
                        Ok(output) => self.spec.outputs.push(Rc::new(output)),
                        Err(e) => error.join(e),
                    }
                },
                Rule::SimpleTrigger => {
                    match self.parse_trigger(pair) {
                        Ok(trigger) => self.spec.outputs.push(Rc::new(trigger)),
                        Err(e) => error.join(e),
                    }
                },
                Rule::TypeDecl => {
                    let type_decl = self.parse_type_declaration(pair);
                    self.spec.type_declarations.push(type_decl);
                },
                Rule::EOI => {},
                _ => unreachable!(),
            }
        }
        Result::from(error)?;
        Ok(self.spec)
    }

    fn parse_import(&self, pair: Pair<Rule>) -> Import {
        assert_eq!(pair.as_rule(), Rule::ImportStmt);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        Import {
            name,
            id: self.spec.next_id(),
            span,
        }
    }

    /**
     * Transforms a `Rule::ConstantStream` into `Constant` AST node.
     * Panics if input is not `Rule::ConstantStream`.
     * The constant rule consists of the following tokens:
     * - `Rule::Ident`
     * - `Rule::Type`
     * - `Rule::Literal`
     */
    fn parse_constant(&self, pair: Pair<'_, Rule>) -> Constant {
        assert_eq!(pair.as_rule(), Rule::ConstantStream);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        let ty = self.parse_type(pairs.next().expect("mismatch between grammar and AST"));
        let literal = self.parse_literal(pairs.next().expect("mismatch between grammar and AST"));
        Constant {
            id: self.spec.next_id(),
            name,
            ty: Some(ty),
            literal,
            span,
        }
    }

    /**
     * Transforms a `Rule::InputStream` into `Input` AST node.
     * Panics if input is not `Rule::InputStream`.
     * The input rule consists of non-empty sequences of following tokens:
     * - `Rule::Ident`
     * - (`Rule::ParamList`)?
     * - `Rule::Type`
     */
    fn parse_inputs(&self, pair: Pair<'_, Rule>) -> Vec<Input> {
        assert_eq!(pair.as_rule(), Rule::InputStream);
        let mut inputs = Vec::new();
        let mut pairs = pair.into_inner();
        while let Some(pair) = pairs.next() {
            let start = pair.as_span().start();
            let name = self.parse_ident(&pair);

            let mut pair = pairs.next().expect("mismatch between grammar and AST");
            let params = if let Rule::ParamList = pair.as_rule() {
                let res = self.parse_parameter_list(pair.into_inner());
                pair = pairs.next().expect("mismatch between grammar and AST");

                res
            } else {
                Vec::new()
            };
            let end = pair.as_span().end();
            let ty = self.parse_type(pair);
            inputs.push(Input {
                id: self.spec.next_id(),
                name,
                params: params.into_iter().map(Rc::new).collect(),
                ty,
                span: Span::Direct { start, end },
            })
        }

        assert!(!inputs.is_empty());
        inputs
    }

    /**
     * Transforms a `Rule::MirrorStream` into `Mirror` AST node.
     * Panics if input is not `Rule::MirrorStream`.
     * The mirror rule consists of the following tokens:
     * - `Rule::Ident` (name)
     * - `Rule::Ident` (target)
     * - `Rule::Expr` (mirror condition)
     */
    fn parse_mirror(&self, pair: Pair<'_, Rule>) -> Result<Mirror, RtLolaError> {
        assert_eq!(pair.as_rule(), Rule::MirrorStream);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        let target = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        let condition =
            self.build_expression_ast(pairs.next().expect("mismatch between grammar and AST").into_inner())?;
        Ok(Mirror {
            name,
            target,
            filter: condition,
            id: self.spec.next_id(),
            span,
        })
    }

    /**
     * Transforms a `Rule::OutputStream` into `Output` AST node.
     * Panics if input is not `Rule::OutputStream`.
     * The output rule consists of the following tokens:
     * - `Rule::Ident`
     * - `Rule::Type`
     * - `Rule::Expr`
     */
    fn parse_output(&self, pair: Pair<'_, Rule>) -> Result<Output, RtLolaError> {
        assert_eq!(pair.as_rule(), Rule::OutputStream);

        let span = pair.as_span().into();
        let mut pairs = pair.into_inner().peekable();

        let pair = pairs.next().unwrap();
        let kind = match pair.as_rule() {
            Rule::TriggerDecl => OutputKind::Trigger,
            Rule::NamedOutputDecl => {
                OutputKind::NamedOutput(
                    self.parse_ident(&pair.into_inner().next().expect("mismatch between grammar and AST")),
                )
            },
            _ => panic!("mismatch between grammar and AST"),
        };

        let mut error = RtLolaError::new();
        let mut eval = Vec::new();
        let mut spawn: Option<SpawnSpec> = None;
        let mut close: Option<CloseSpec> = None;

        let mut pair = pairs.peek().expect("mismatch between grammar and AST");
        let params = if let Rule::ParamList = pair.as_rule() {
            let local_pair = pairs.next().expect("mismatch between grammar and AST");
            let res = self.parse_parameter_list(local_pair.into_inner());

            res
        } else {
            Vec::new()
        };

        pair = pairs.peek().expect("mismatch between grammar and AST");
        let annotated_type = if let Rule::Type = pair.as_rule() {
            let local_pair = pairs.next().expect("mismatch between grammar and AST");
            let ty = self.parse_type(local_pair);

            Some(ty)
        } else {
            None
        };

        pairs.for_each(|pair| {
            match pair.as_rule() {
                Rule::SpawnDecl => {
                    if let Some(old_spawn) = &spawn {
                        let err = Diagnostic::error("Multiple Spawn clauses found")
                            .add_span_with_label(old_spawn.span, Some("first Spawn here"), true)
                            .add_span_with_label(pair.as_span().into(), Some("Second Spawn clause found here"), false);
                        error.add(err);
                    }
                    let spawn_spec = self.parse_spawn_spec(pair);
                    spawn = spawn_spec.map_or_else(
                        |e| {
                            error.join(e);
                            None
                        },
                        Some,
                    )
                },
                Rule::CloseDecl => {
                    if let Some(old_close) = &close {
                        let err = Diagnostic::error("Multiple Close clauses found")
                            .add_span_with_label(old_close.span, Some("first Close here"), true)
                            .add_span_with_label(pair.as_span().into(), Some("Second Close clause found here"), false);
                        error.add(err);
                    }
                    let close_spec = self.parse_close_spec(pair);
                    close = close_spec.map_or_else(
                        |e| {
                            error.join(e);
                            None
                        },
                        Some,
                    );
                },
                Rule::EvalDecl => {
                    let eval_spec = self.parse_eval_spec(pair);
                    match eval_spec {
                        Ok(eval_spec) => eval.push(eval_spec),
                        Err(e) => error.join(e),
                    }
                },
                Rule::SimpleEvalDecl => {
                    let eval_spec = self.parse_eval_spec_simple(pair.clone());
                    match eval_spec {
                        Ok(eval_spec) => {
                            debug_assert!(eval.is_empty(), "must be empty due to grammar restrictions");
                            eval.push(eval_spec)
                        },
                        Err(e) => error.join(e),
                    }
                },
                _ => {
                    unreachable!("mismatch between grammar and AST")
                },
            }
        });

        let eval = eval
            .into_iter()
            .map(|mut eval| {
                // if the output is a trigger, a missing eval-with clause is replaced with an empty trigger message
                // if the output is a named output, a missing eval-with clause is replaced by an empty tuple
                eval.eval_expression.get_or_insert_with(|| {
                    let kind = match kind {
                        OutputKind::Trigger => {
                            ExpressionKind::Lit(Literal {
                                kind: LitKind::Str("".into()),
                                id: self.spec.next_id(),
                                span: Span::Unknown,
                            })
                        },
                        OutputKind::NamedOutput(_) => ExpressionKind::Tuple(vec![]),
                    };
                    Expression {
                        kind,
                        id: self.spec.next_id(),
                        span: Span::Unknown,
                    }
                });
                eval
            })
            .collect();

        Result::from(error)?;
        Ok(Output {
            id: self.spec.next_id(),
            kind,
            annotated_type,
            params: params.into_iter().map(Rc::new).collect(),
            spawn,
            eval,
            close,
            span,
        })
    }

    fn parse_parameter_list(&self, param_list: Pairs<'_, Rule>) -> Vec<Parameter> {
        let mut params = Vec::new();
        for (ix, param_decl) in param_list.enumerate() {
            assert_eq!(Rule::ParameterDecl, param_decl.as_rule());
            let span = param_decl.as_span().into();
            let mut decl = param_decl.into_inner();
            let name = self.parse_ident(&decl.next().expect("mismatch between grammar and AST"));
            let ty = if let Some(type_pair) = decl.next() {
                assert_eq!(Rule::Type, type_pair.as_rule());
                Some(self.parse_type(type_pair))
            } else {
                None
            };
            params.push(Parameter {
                name,
                ty,
                param_idx: ix,
                id: self.spec.next_id(),
                span,
            });
        }
        params
    }

    fn parse_spawn_spec(&self, spawn_pair: Pair<'_, Rule>) -> Result<SpawnSpec, RtLolaError> {
        let span_inv: Span = spawn_pair.as_span().into();

        let mut spawn_children = spawn_pair.into_inner().peekable();
        let next_pair = spawn_children.peek();

        let mut error = RtLolaError::new();

        let annotated_pacing = if let Some(pair) = next_pair {
            if let Rule::ActivationCondition = pair.as_rule() {
                let expr = self.build_expression_ast(pair.clone().into_inner());
                spawn_children.next();
                expr.map_or_else(
                    |e| {
                        error.join(e);
                        None
                    },
                    Some,
                )
            } else {
                None
            }
        } else {
            None
        };

        let mut condition: Option<Expression> = None;
        let mut expression: Option<Expression> = None;
        for pair in spawn_children {
            match pair.as_rule() {
                Rule::SpawnWhen => {
                    if let Some(old_condition) = &condition {
                        let err = Diagnostic::error("Multiple Spawn conditions found")
                            .add_span_with_label(old_condition.span, Some("first spawn condition here"), true)
                            .add_span_with_label(pair.as_span().into(), Some("Second condition found here"), false);
                        error.add(err);
                    }
                    let condition_pair = pair.into_inner().next().expect("mismatch between grammar and AST");
                    let condition_exp = self.build_expression_ast(condition_pair.into_inner());
                    condition = condition_exp.map_or_else(
                        |e| {
                            error.join(e);
                            None
                        },
                        Some,
                    )
                },
                Rule::SpawnWith => {
                    if let Some(old_expression) = &expression {
                        let err = Diagnostic::error("Multiple Spawn expressions found")
                            .add_span_with_label(old_expression.span, Some("first spawn expression here"), true)
                            .add_span_with_label(pair.as_span().into(), Some("Second expression found here"), false);
                        error.add(err);
                    }
                    let expression_pair = pair.into_inner().next().expect("mismatch between grammar and AST");
                    let spawn_expr = self.build_expression_ast(expression_pair.into_inner());
                    expression = spawn_expr.map_or_else(
                        |e| {
                            error.join(e);
                            None
                        },
                        Some,
                    )
                },
                _ => unreachable!("mismatch between grammar and AST"),
            }
        }

        if expression.is_none() && condition.is_none() && annotated_pacing.is_none() {
            error.add(
                Diagnostic::error("Spawn clause needs a condition, expression or pacing").add_span_with_label(
                    span_inv,
                    Some("found spawn here"),
                    true,
                ),
            );
        }

        Result::from(error)?;

        Ok(SpawnSpec {
            expression,
            annotated_pacing,
            condition,
            id: self.spec.next_id(),
            span: span_inv,
        })
    }

    fn parse_eval_spec_simple(&self, ext_pair: Pair<'_, Rule>) -> Result<EvalSpec, RtLolaError> {
        let span_ext: Span = ext_pair.as_span().into();

        let mut children = ext_pair.into_inner();
        let mut next_pair = children.next();

        let annotated_pacing = if let Some(pair) = next_pair.clone() {
            if let Rule::ActivationCondition = pair.as_rule() {
                let expr = self.build_expression_ast(pair.into_inner())?;
                next_pair = children.next();
                Some(expr)
            } else {
                None
            }
        } else {
            None
        };

        let exp_res = self.build_expression_ast(next_pair.expect("Mismatch between grammar and AST").into_inner())?;

        Ok(EvalSpec {
            annotated_pacing,
            condition: None,
            eval_expression: Some(exp_res),
            id: self.spec.next_id(),
            span: span_ext,
        })
    }

    fn parse_eval_spec(&self, ext_pair: Pair<'_, Rule>) -> Result<EvalSpec, RtLolaError> {
        let span_ext: Span = ext_pair.as_span().into();

        let mut children = ext_pair.into_inner().peekable();
        let next_pair = children.peek();

        let mut error = RtLolaError::new();

        let annotated_pacing = if let Some(pair) = next_pair {
            if let Rule::ActivationCondition = pair.as_rule() {
                let expr = self.build_expression_ast(pair.clone().into_inner())?;
                children.next();
                Some(expr)
            } else {
                None
            }
        } else {
            None
        };

        let mut condition: Option<Expression> = None;
        let mut eval_expr: Option<Expression> = None;
        for pair in children {
            match pair.as_rule() {
                Rule::EvalWhen => {
                    if let Some(old_cond) = &condition {
                        let err = Diagnostic::error("Multiple evaluation conditions found")
                            .add_span_with_label(old_cond.span, Some("first condition here"), true)
                            .add_span_with_label(pair.as_span().into(), Some("Second condition found here"), false);
                        error.add(err);
                    }
                    let condition_pair = pair.into_inner().next().expect("mismatch between grammar and AST");
                    let cond_expr = self.build_expression_ast(condition_pair.into_inner());
                    condition = cond_expr.map_or_else(
                        |e| {
                            error.join(e);
                            None
                        },
                        Some,
                    )
                },
                Rule::EvalWith => {
                    if let Some(old_eval) = &eval_expr {
                        let err = Diagnostic::error("Multiple eval expressions found")
                            .add_span_with_label(old_eval.span, Some("first eval expression here"), true)
                            .add_span_with_label(pair.as_span().into(), Some("Second expression found here"), false);
                        error.add(err);
                    }
                    let expression_pair = pair.into_inner().next().expect("mismatch between grammar and AST");
                    let target_exp = self.build_expression_ast(expression_pair.into_inner());
                    eval_expr = target_exp.map_or_else(
                        |e| {
                            error.join(e);
                            None
                        },
                        Some,
                    )
                },
                _ => unreachable!("mismatch between grammar and AST"),
            }
        }

        if eval_expr.is_none() && condition.is_none() && annotated_pacing.is_none() {
            error.add(
                Diagnostic::error("Eval clause needs either expression or condition").add_span_with_label(
                    span_ext,
                    Some("found eval clause here"),
                    true,
                ),
            );
        }

        Result::from(error)?;

        Ok(EvalSpec {
            annotated_pacing,
            condition,
            eval_expression: eval_expr,
            id: self.spec.next_id(),
            span: span_ext,
        })
    }

    fn parse_close_spec(&self, close_pair: Pair<'_, Rule>) -> Result<CloseSpec, RtLolaError> {
        let span_close: Span = close_pair.as_span().into();

        let mut children = close_pair.into_inner();

        let mut next_pair = children.next();

        let annotated_pacing = if let Some(pair) = next_pair.clone() {
            if let Rule::ActivationCondition = pair.as_rule() {
                let expr = self.build_expression_ast(pair.into_inner())?;
                next_pair = children.next();
                Some(expr)
            } else {
                None
            }
        } else {
            None
        };

        let condition_pair = next_pair.expect("mismatch between grammar and ast");
        let condition = match condition_pair.as_rule() {
            Rule::Expr => self.build_expression_ast(condition_pair.into_inner()),
            _ => unreachable!(),
        }?;
        Ok(CloseSpec {
            condition,
            annotated_pacing,
            id: self.spec.next_id(),
            span: span_close,
        })
    }

    /**
     * Transforms a `Rule::Trigger` into `Trigger` AST node.
     * Panics if input is not `Rule::Trigger`.
     * The output rule consists of the following tokens:
     * - (`Rule::Ident`)?
     * - `Rule::Expr`
     * - (`Rule::StringLiteral`)?
     */
    fn parse_trigger(&self, pair: Pair<'_, Rule>) -> Result<Output, RtLolaError> {
        assert_eq!(pair.as_rule(), Rule::SimpleTrigger);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();

        let mut pair = pairs.next().expect("mismatch between grammar and AST");

        // Parse the `@ [Expr]` part of output declaration
        let annotated_pacing_type = if let Rule::ActivationCondition = pair.as_rule() {
            let expr = self.build_expression_ast(pair.into_inner())?;
            pair = pairs.next().expect("mismatch between grammar and AST");
            Some(expr)
        } else {
            None
        };

        let expression = self.build_expression_ast(pair.into_inner())?;

        let (msg, msg_span) = pairs
            .next()
            .map(|pair| {
                assert_eq!(pair.as_rule(), Rule::String);
                (pair.as_str().to_owned(), pair.as_span().into())
            })
            .unwrap_or(("".into(), Span::Unknown));

        let msg_expr = Expression {
            kind: ExpressionKind::Lit(Literal::new_str(self.spec.next_id(), &msg, span)),
            id: self.spec.next_id(),
            span: msg_span,
        };

        Ok(Output {
            kind: OutputKind::Trigger,
            annotated_type: None,
            params: Vec::new(),
            spawn: None,
            eval: vec![EvalSpec {
                annotated_pacing: annotated_pacing_type,
                condition: Some(expression),
                eval_expression: Some(msg_expr),
                id: self.spec.next_id(),
                span,
            }],
            close: None,
            id: self.spec.next_id(),
            span,
        })
    }

    /**
     * Transforms a `Rule::Ident` into `Ident` AST node.
     * Panics if input is not `Rule::Ident`.
     */
    fn parse_ident(&self, pair: &Pair<'_, Rule>) -> Ident {
        assert_eq!(pair.as_rule(), Rule::Ident);
        let name = pair.as_str().to_string();
        Ident::new(name, pair.as_span().into())
    }

    /**
     * Transforms a `Rule::TypeDecl` into `TypeDeclaration` AST node.
     * Panics if input is not `Rule::TypeDecl`.
     */
    fn parse_type_declaration(&self, pair: Pair<'_, Rule>) -> TypeDeclaration {
        assert_eq!(pair.as_rule(), Rule::TypeDecl);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        let mut fields = Vec::new();
        while let Some(pair) = pairs.next() {
            let field_name = pair.as_str().to_string();
            let ty = self.parse_type(pairs.next().expect("mismatch between grammar and AST"));
            fields.push(Box::new(TypeDeclField {
                name: field_name,
                ty,
                id: self.spec.next_id(),
                span: pair.as_span().into(),
            }));
        }

        TypeDeclaration {
            name: Some(name),
            span,
            id: self.spec.next_id(),
            fields,
        }
    }

    /**
     * Transforms a `Rule::Type` into `Type` AST node.
     * Panics if input is not `Rule::Type`.
     */
    fn parse_type(&self, pair: Pair<'_, Rule>) -> Type {
        assert_eq!(pair.as_rule(), Rule::Type);
        let span = pair.as_span();
        let mut tuple = Vec::new();
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Ident => {
                    return Type::new_simple(self.spec.next_id(), pair.as_str().to_string(), pair.as_span().into());
                },
                Rule::Type => tuple.push(self.parse_type(pair)),
                Rule::Optional => {
                    let span = pair.as_span();
                    let inner = pair
                        .into_inner()
                        .next()
                        .expect("mismatch between grammar and AST: first argument is a type");
                    let inner_ty =
                        Type::new_simple(self.spec.next_id(), inner.as_str().to_string(), inner.as_span().into());
                    return Type::new_optional(self.spec.next_id(), inner_ty, span.into());
                },
                _ => unreachable!("{:?} is not a type, ensured by grammar", pair.as_rule()),
            }
        }
        Type::new_tuple(self.spec.next_id(), tuple, span.into())
    }

    /**
     * Transforms a `Rule::Literal` into `Literal` AST node.
     * Panics if input is not `Rule::Literal`.
     */
    fn parse_literal(&self, pair: Pair<'_, Rule>) -> Literal {
        assert_eq!(pair.as_rule(), Rule::Literal);
        let inner = pair.into_inner().next().expect("Rule::Literal has exactly one child");
        match inner.as_rule() {
            Rule::String => {
                let str_rep = inner.as_str();
                Literal::new_str(self.spec.next_id(), str_rep, inner.as_span().into())
            },
            Rule::RawString => {
                let str_rep = inner.as_str();
                Literal::new_raw_str(self.spec.next_id(), str_rep, inner.as_span().into())
            },
            Rule::NumberLiteral => {
                let span = inner.as_span();
                let mut pairs = inner.into_inner();
                let value = pairs.next().expect("Mismatch between AST and grammar");

                let str_rep: &str = value.as_str();
                let unit = pairs.next().map(|unit| unit.as_str().to_string());

                Literal::new_numeric(self.spec.next_id(), str_rep, unit, span.into())
            },
            Rule::True => Literal::new_bool(self.spec.next_id(), true, inner.as_span().into()),
            Rule::False => Literal::new_bool(self.spec.next_id(), false, inner.as_span().into()),
            _ => unreachable!(),
        }
    }

    #[allow(clippy::vec_box)]
    fn parse_vec_of_expressions(&self, pairs: Pairs<'_, Rule>) -> Result<Vec<Expression>, RtLolaError> {
        type ExprRes = Result<Expression, RtLolaError>;
        let (exprs, errs): (Vec<ExprRes>, Vec<ExprRes>) = pairs
            .map(|expr| self.build_expression_ast(expr.into_inner()))
            .partition(Result::is_ok);
        Result::from(errs.into_iter().flat_map(Result::unwrap_err).collect::<RtLolaError>())?;
        Ok(exprs.into_iter().map(Result::unwrap).collect())
    }

    fn parse_vec_of_types(&self, pairs: Pairs<'_, Rule>) -> Vec<Type> {
        pairs.map(|p| self.parse_type(p)).collect()
    }

    fn build_function_expression(&self, pair: Pair<'_, Rule>, span: Span) -> Result<Expression, RtLolaError> {
        let mut children = pair.into_inner();
        let fun_name = self.parse_ident(&children.next().unwrap());
        let mut next = children.next().expect("Mismatch between AST and parser");
        let type_params = match next.as_rule() {
            Rule::GenericParam => {
                let params = self.parse_vec_of_types(next.into_inner());
                next = children.next().expect("Mismatch between AST and parser");
                params
            },
            Rule::FunctionArgs => Vec::new(),
            _ => unreachable!(),
        };
        assert_eq!(next.as_rule(), Rule::FunctionArgs);
        let mut args = Vec::new();
        let mut error = RtLolaError::new();
        let mut arg_names = Vec::new();
        for pair in next.into_inner() {
            assert_eq!(pair.as_rule(), Rule::FunctionArg);
            let mut pairs = pair.into_inner();
            let mut pair = pairs.next().expect("Mismatch between AST and parser");
            if pair.as_rule() == Rule::Ident {
                // named argument
                arg_names.push(Some(self.parse_ident(&pair)));
                pair = pairs.next().expect("Mismatch between AST and parser");
            } else {
                arg_names.push(None);
            }
            match self.build_expression_ast(pair.into_inner()) {
                Ok(e) => args.push(e),
                Err(e) => error.join(e),
            }
        }
        Result::from(error)?;
        let name = FunctionName {
            name: fun_name,
            arg_names,
        };
        Ok(Expression::new(
            self.spec.next_id(),
            ExpressionKind::Function(name, type_params, args),
            span,
        ))
    }

    /**
     * Builds the Expr AST.
     */
    fn build_expression_ast(&self, pairs: Pairs<'_, Rule>) -> Result<Expression, RtLolaError> {
        PRATT_PARSER
            .map_primary(|primary| self.build_term_ast(primary))
            .map_infix(|lhs, op, rhs| {

                // Reduce function combining `Expression`s to `Expression`s with the correct precs
                let (lhs, rhs) = RtLolaError::combine(lhs, rhs, |a, b| (a,b))?;
                let span = lhs.span.union(&rhs.span);
                let op = match op.as_rule() {
                    // Arithmetic
                    Rule::Add => BinOp::Add,
                    Rule::Subtract => BinOp::Sub,
                    Rule::Multiply => BinOp::Mul,
                    Rule::Divide => BinOp::Div,
                    Rule::Mod => BinOp::Rem,
                    Rule::Power => BinOp::Pow,
                    // Logical
                    Rule::And => BinOp::And,
                    Rule::Or => BinOp::Or,
                    Rule::Implies => BinOp::Implies,
                    // Comparison
                    Rule::LessThan => BinOp::Lt,
                    Rule::LessThanOrEqual => BinOp::Le,
                    Rule::MoreThan => BinOp::Gt,
                    Rule::MoreThanOrEqual => BinOp::Ge,
                    Rule::Equal => BinOp::Eq,
                    Rule::NotEqual => BinOp::Ne,
                    // Bitwise
                    Rule::BitAnd => BinOp::BitAnd,
                    Rule::BitOr => BinOp::BitOr,
                    Rule::BitXor => BinOp::BitXor,
                    Rule::ShiftLeft => BinOp::Shl,
                    Rule::ShiftRight => BinOp::Shr,
                    // bubble up the unary operator on the lhs (if it exists) to fix precedence
                    Rule::Dot => {
                        let (unop, binop_span, inner) = match lhs.kind {
                            ExpressionKind::Unary(unop, inner) => (Some(unop), inner.span.union(&rhs.span), inner),
                            _ => (None, span, Box::new(lhs)),
                        };
                        match rhs.kind {
                            // access to a tuple
                            ExpressionKind::Lit(l) => {
                                let ident = match l.kind {
                                    LitKind::Numeric(val, unit) => {
                                        assert!(unit.is_none());
                                        Ident::new(val, l.span)
                                    }
                                    _ => {
                                        return Err(Diagnostic::error(&format!("expected unsigned integer, found {l}")).add_span_with_label(rhs.span, Some("unexpected"), true).into());
                                    }
                                };
                                let binop_expr =
                                    Expression::new(self.spec.next_id(), ExpressionKind::Field(inner, ident), binop_span);
                                match unop {
                                    None => return Ok(binop_expr),
                                    Some(unop) => {
                                        return Ok(Expression::new(
                                            self.spec.next_id(),
                                            ExpressionKind::Unary(unop, Box::new(binop_expr)),
                                            span,
                                        ))
                                    }
                                }
                            }
                            ExpressionKind::Function(name, types, args) => {
                                // match for builtin function names and transform them into appropriate AST nodes
                                let signature = name.to_string();
                                let kind = match signature.as_str() {
                                    "defaults(to:)" => {
                                        assert_eq!(args.len(), 1);
                                        ExpressionKind::Default(inner, Box::new(args[0].clone()))
                                    }
                                    "offset(by:)" => {
                                        assert_eq!(args.len(), 1);
                                        let offset_expr = &args[0];
                                        let rhs_span = rhs.span;
                                        let offset = offset_expr.parse_offset().map_err(|reason| Diagnostic::error("failed to parse offset").add_span_with_label(rhs_span, Some(&reason), true))?;

                                        ExpressionKind::Offset(inner, offset)
                                    }
                                    "hold()" => {
                                        assert_eq!(args.len(), 0);
                                        ExpressionKind::StreamAccess(inner, StreamAccessKind::Hold)
                                    }
                                    "hold(or:)" => {
                                        assert_eq!(args.len(), 1);
                                        let lhs = Expression::new(
                                            self.spec.next_id(),
                                            ExpressionKind::StreamAccess(inner, StreamAccessKind::Hold),
                                            span,
                                        );
                                        ExpressionKind::Default(Box::new(lhs), Box::new(args[0].clone()))
                                    }
                                    "get()" => {
                                        assert_eq!(args.len(), 0);
                                        ExpressionKind::StreamAccess(inner, StreamAccessKind::Get)
                                    }
                                    "get(or:)" => {
                                        assert_eq!(args.len(), 1);
                                        let lhs = Expression::new(
                                            self.spec.next_id(),
                                            ExpressionKind::StreamAccess(inner, StreamAccessKind::Get),
                                            span,
                                        );
                                        ExpressionKind::Default(Box::new(lhs), Box::new(args[0].clone()))
                                    }
                                    "is_fresh()" => {
                                        assert_eq!(args.len(), 0);
                                        ExpressionKind::StreamAccess(inner, StreamAccessKind::Fresh)
                                    }
                                    "aggregate(over_discrete:using:)" | "aggregate(over_exactly_discrete:using:)" |"aggregate(over:using:)" | "aggregate(over_exactly:using:)" | "aggregate(over_instances:using:)"=> {
                                        assert_eq!(args.len(), 2);
                                        let window_op = match &args[1].kind {
                                            ExpressionKind::Ident(i) => match i.name.as_str() {
                                                "Σ" | "sum" => WindowOperation::Sum,
                                                "#" | "count" => WindowOperation::Count,
                                                //"Π" | "prod" | "product" => WindowOperation::Product,
                                                "∫" | "integral" => WindowOperation::Integral,
                                                "avg" | "average" => WindowOperation::Average,
                                                "min" => WindowOperation::Min,
                                                "max" => WindowOperation::Max,
                                                "∃" | "disjunction" | "∨" | "exists" => {
                                                    WindowOperation::Disjunction
                                                }
                                                "∀" | "conjunction" | "∧" | "forall" => {
                                                    WindowOperation::Conjunction
                                                }
                                                "last" => WindowOperation::Last,
                                                "variance" | "var" | "σ²" => WindowOperation::Variance,
                                                "covariance" | "cov" => WindowOperation::Covariance,
                                                "standard_deviation" | "sd" | "σ" => WindowOperation::StandardDeviation,
                                                "median" | "med" | "µ" => WindowOperation::NthPercentile(50),
                                                _ if i.name.as_str().starts_with("pctl") &&
                                                    i.name.as_str().chars().skip("pctl".len()).all(|c| c.is_numeric()) => {
                                                    let n_string = i.name.as_str().to_string();
                                                    let n_string: String = n_string.chars().skip("pctl".len()).collect();
                                                    let percentile: usize = n_string.parse::<usize>().map_err(|_|
                                                        RtLolaError::from(Diagnostic::error(&format!("unknown aggregation function {}, invalid number-percentile suffix {}", i.name, n_string)).add_span_with_label(i.span, Some("available: count, min, max, sum, average, exists, forall, integral, last, variance, covariance, standard_deviation, median, pctlX with 0 ≤ X ≤ 100 (e.g. pctl25)"), true))
                                                    )?;
                                                    if percentile > 100{
                                                        return Err(Diagnostic::error(&format!("unknown aggregation function {}, invalid percentile suffix", i.name)).add_span_with_label( i.span, Some("available: count, min, max, sum, average, exists, forall, integral, last, variance, covariance, standard_deviation, median, pctlX with 0 ≤ X ≤ 100 (e.g. pctl25)"), true).into());

                                                    }
                                                    WindowOperation::NthPercentile(percentile as u8)
                                                }
                                                fun => {
                                                    return Err(Diagnostic::error(&format!("unknown aggregation function {fun}")).add_span_with_label(i.span, Some("available: count, min, max, sum, average, exists, forall, integral, last, variance, covariance, standard_deviation, median, pctlX with 0 ≤ X ≤ 100 (e.g. pctl25)"), true).into());
                                                }
                                            },
                                            _ => {
                                                return Err(Diagnostic::error("expected aggregation function").add_span_with_label(args[1].span, Some("available: count, min, max, sum, average, exists, forall, integral, last, variance, covariance, standard_deviation, median, pctlX with 0 ≤ X ≤ 100 (e.g. pctl25)"), true).into());
                                            }
                                        };
                                        if signature.contains("instances") {
                                            let instances = match &args[0].kind {
                                                ExpressionKind::Ident(i) => match i.name.as_str() {
                                                    "fresh" | "Fresh" => InstanceSelection::Fresh,
                                                    "all" | "All" => InstanceSelection::All,
                                                    sel => {
                                                        return Err(Diagnostic::error(&format!("unknown instance selection {sel}")).add_span_with_label(i.span, Some("available: fresh, all"), true).into());
                                                    }
                                                }
                                                _ => {
                                                    return Err(Diagnostic::error("expected instance selection").add_span_with_label(args[0].span, Some("available: fresh, all"), true).into());
                                                }
                                            };
                                            let aggregation = window_op.try_into().map_err(|reason| Diagnostic::error(&format!("Operation not supported: {reason}")).add_span_with_label(args[1].span, Some("available: count, min, max, sum, average, exists, forall, variance, covariance, standard_deviation, median, pctlX with 0 ≤ X ≤ 100 (e.g. pctl25)"), true))?;
                                            ExpressionKind::InstanceAggregation { expr: inner, selection: instances, aggregation }
                                        }
                                        else if signature.contains("discrete") {
                                            if window_op == WindowOperation::Last {
                                                // Todo: This should be a warning
                                                // return Err(Diagnostic::error("discrete window operation: last has same semantics as .offset(by:-1) and is more expensive").add_span_with_label(args[1].span.clone(), Some("don't use last for discrete windows"), true).into());
                                            }
                                            ExpressionKind::DiscreteWindowAggregation {
                                                expr: inner,
                                                duration: Box::new(args[0].clone()),
                                                wait: signature.contains("over_exactly"),
                                                aggregation: window_op,
                                            }
                                        } else {
                                            ExpressionKind::SlidingWindowAggregation {
                                                expr: inner,
                                                duration: Box::new(args[0].clone()),
                                                wait: signature.contains("over_exactly"),
                                                aggregation: window_op,
                                            }
                                        }
                                    }
                                    _ => ExpressionKind::Method(inner, name, types, args),
                                };
                                let binop_expr = Expression::new(self.spec.next_id(), kind, binop_span);
                                match unop {
                                    None => return Ok(binop_expr),
                                    Some(unop) => {
                                        return Ok(Expression::new(
                                            self.spec.next_id(),
                                            ExpressionKind::Unary(unop, Box::new(binop_expr)),
                                            span,
                                        ))
                                    }
                                }
                            }
                            _ => {
                                return Err(Diagnostic::error(&format!("expected method call or tuple access, found {rhs}")).add_span_with_label(rhs.span, Some("unexpected"), true).into());
                            }
                        }
                    }
                    Rule::OpeningBracket => {
                        let rhs_span = rhs.span;
                        let offset = rhs.parse_offset().map_err(|reason| Diagnostic::error("failed to parse offset expression").add_span_with_label(rhs_span, Some(&reason), true))?;
                        match lhs.kind {
                            ExpressionKind::Unary(unop, inner) => {
                                let inner_span = inner.span.union(&rhs.span);
                                let new_inner =
                                    Expression::new(self.spec.next_id(), ExpressionKind::Offset(inner, offset), inner_span);
                                return Ok(Expression::new(
                                    self.spec.next_id(),
                                    ExpressionKind::Unary(unop, Box::new(new_inner)),
                                    span,
                                ));
                            }
                            _ => {
                                return Ok(Expression::new(
                                    self.spec.next_id(),
                                    ExpressionKind::Offset(lhs.into(), offset),
                                    span,
                                ))
                            }
                        }
                    }
                    _ => unreachable!(),
                };
                Ok(Expression::new(self.spec.next_id(), ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)), span))
            }).parse(pairs)
    }

    fn build_term_ast(&self, pair: Pair<'_, Rule>) -> Result<Expression, RtLolaError> {
        let span = pair.as_span();
        match pair.as_rule() {
            // Map function from `Pair` to AST data structure `Expression`
            Rule::Literal => {
                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::Lit(self.parse_literal(pair)),
                    span.into(),
                ))
            },
            Rule::Ident => {
                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::Ident(self.parse_ident(&pair)),
                    span.into(),
                ))
            },
            Rule::ParenthesizedExpression => {
                let mut inner = pair.into_inner();
                let opp = inner.next().expect(
                    "Rule::ParenthesizedExpression has a token for the (potentialy missing) opening parenthesis",
                );
                let opening_parenthesis = if let Rule::OpeningParenthesis = opp.as_rule() {
                    Some(Box::new(Parenthesis::new(self.spec.next_id(), opp.as_span().into())))
                } else {
                    None
                };

                let inner_expression = inner
                    .next()
                    .expect("Rule::ParenthesizedExpression has a token for the contained expression");

                let closing = inner.next().expect(
                    "Rule::ParenthesizedExpression has a token for the (potentialy missing) closing parenthesis",
                );
                let closing_parenthesis = if let Rule::ClosingParenthesis = closing.as_rule() {
                    Some(Box::new(Parenthesis::new(
                        self.spec.next_id(),
                        closing.as_span().into(),
                    )))
                } else {
                    None
                };

                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::ParenthesizedExpression(
                        opening_parenthesis,
                        Box::new(self.build_expression_ast(inner_expression.into_inner())?),
                        closing_parenthesis,
                    ),
                    span.into(),
                ))
            },
            Rule::UnaryExpr => {
                // First child is the operator, second the operand.
                let mut children = pair.into_inner();
                let pest_operator = children.next().expect("Unary expressions need to have an operator.");
                let operand = children.next().expect("Unary expressions need to have an operand.");
                let operand = self.build_term_ast(operand)?;
                let operator = match pest_operator.as_rule() {
                    Rule::Add => return Ok(operand), // Discard unary plus because it is semantically null.
                    Rule::Subtract => UnOp::Neg,
                    Rule::Neg => UnOp::Not,
                    Rule::BitNot => UnOp::BitNot,
                    _ => unreachable!(),
                };
                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::Unary(operator, Box::new(operand)),
                    span.into(),
                ))
            },
            Rule::TernaryExpr => {
                let mut children = self.parse_vec_of_expressions(pair.into_inner())?;
                assert_eq!(children.len(), 3, "A ternary expression needs exactly three children.");
                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::Ite(
                        Box::new(children.remove(0)),
                        Box::new(children.remove(0)),
                        Box::new(children.remove(0)),
                    ),
                    span.into(),
                ))
            },
            Rule::Tuple => {
                let elements = self.parse_vec_of_expressions(pair.into_inner())?;
                assert!(elements.len() != 1, "Tuples may not have exactly one element.");
                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::Tuple(elements),
                    span.into(),
                ))
            },
            Rule::Expr => self.build_expression_ast(pair.into_inner()),
            Rule::FunctionExpr => self.build_function_expression(pair, span.into()),
            Rule::IntegerLiteral => {
                let span: Span = span.into();
                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::Lit(Literal::new_numeric(self.spec.next_id(), pair.as_str(), None, span)),
                    span,
                ))
            },
            Rule::MissingExpression => {
                let span = span.into();
                Ok(Expression::new(
                    self.spec.next_id(),
                    ExpressionKind::MissingExpression,
                    span,
                ))
            },
            _ => unreachable!("Unexpected rule when parsing expression ast: {:?}", pair.as_rule()),
        }
    }

    pub(crate) fn parse_rational(repr: &str) -> Result<Rational, String> {
        // precondition: repr is a valid floating point literal
        debug_assert!(repr.parse::<f64>().is_ok());

        macro_rules! split_at {
            ($s:expr, $c:literal) => {{
                let (prefix, suffix) = $s.split_at($s.find($c).unwrap_or($s.len()));
                let suffix = if suffix.len() > 0 { &suffix[1..] } else { suffix };
                (prefix, suffix)
            }};
        }

        let (int_digits, suffix) = split_at!(repr, '.'); // actually sign + int_digits
        let (dec_digits, exp_str) = split_at!(suffix, 'e');

        let digits = int_digits.to_string() + dec_digits; // actually sign + digits
        let integer = match BigInt::from_str(digits.as_str()) {
            Ok(i) => i,
            Err(e) => return Err(format!("parsing rational '{repr}' failed: {e}")),
        };
        let mut r = BigRational::from(integer);
        if !dec_digits.is_empty() {
            // divide by 10 for each decimal place
            r /= BigInt::from_u8(10).unwrap().pow(dec_digits.len());
        }

        if !exp_str.is_empty() {
            let exp = match BigInt::from_str(exp_str) {
                Ok(i) => i,
                Err(e) => return Err(format!("parsing rational '{repr}' failed: {e}")),
            };
            let exp = match exp.to_i16() {
                Some(i) => i,
                None => {
                    return Err(format!(
                        "parsing rational '{repr}' failed: e exponent {exp} does not fit into i16"
                    ))
                },
            };
            let factor = BigInt::from_u8(10).unwrap().pow(exp.unsigned_abs());
            if exp.is_negative() {
                r /= factor;
            } else {
                r *= factor;
            }
        }

        let p = match (r.numer().to_i64(), r.denom().to_i64()) {
            (Some(n), Some(d)) => (n, d),
            _ => {
                return Err(format!(
                    "parsing rational failed: rational {r} does not fit into Rational64"
                ))
            },
        };
        Ok(Rational::from(p))
    }
}

pub(crate) fn to_rtlola_error(err: pest::error::Error<Rule>) -> RtLolaError {
    use pest::error::*;
    let msg = match err.variant {
        ErrorVariant::ParsingError {
            positives: pos,
            negatives: neg,
        } => {
            match (neg.is_empty(), pos.is_empty()) {
                (false, false) => {
                    format!(
                        "unexpected {}; expected {}",
                        neg.iter().map(|r| format!("{r:?}")).collect::<Vec<String>>().join(", "),
                        pos.iter().map(|r| format!("{r:?}")).collect::<Vec<String>>().join(", ")
                    )
                },
                (false, true) => {
                    format!(
                        "unexpected {}",
                        neg.iter().map(|r| format!("{r:?}")).collect::<Vec<String>>().join(", ")
                    )
                },
                (true, false) => {
                    format!(
                        "expected {}",
                        pos.iter().map(|r| format!("{r:?}")).collect::<Vec<String>>().join(", ")
                    )
                },
                (true, true) => "unknown parsing error".to_owned(),
            }
        },
        ErrorVariant::CustomError { message: msg } => msg,
    };
    let span = match err.location {
        InputLocation::Pos(start) => rtlola_reporting::Span::Direct { start, end: start },
        InputLocation::Span(s) => rtlola_reporting::Span::Direct { start: s.0, end: s.1 },
    };
    Diagnostic::error(&msg)
        .add_span_with_label(span, Some("here"), true)
        .into()
}

#[cfg(test)]
mod tests {
    use pest::{consumes_to, parses_to};

    use super::*;

    fn parse(spec: &str) -> RtLolaAst {
        super::super::parse(&ParserConfig::for_string(spec.into())).unwrap_or_else(|e| panic!("{:?}", e))
    }

    fn parse_without_desugar(spec: &str) -> RtLolaAst {
        let cfg = ParserConfig::for_string(spec.into());
        RtLolaParser::new(&cfg)
            .parse_spec()
            .unwrap_or_else(|e| panic!("{:?}", e))
    }

    fn cmp_ast_spec(ast: &RtLolaAst, spec: &str) -> bool {
        // Todo: Make more robust, e.g. against changes in whitespace.
        assert_eq!(format!("{}", ast), spec);
        true
    }

    #[test]
    fn parse_simple() {
        let _ = LolaParser::parse(Rule::Spec, "input in: Int\noutput out: Int := in\ntrigger in ≠ out")
            .unwrap_or_else(|e| panic!("{}", e));
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
                        NumberLiteral(23, 24, [
                            NumberLiteralValue(23, 24, [])
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_constant_ast() {
        let spec = "constant five : Int := 5";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::ConstantStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_constant(pair);
        assert_eq!(format!("{}", ast), "constant five: Int := 5")
    }

    #[test]
    fn parse_constant_double() {
        let spec = "constant fiveoh: Double := 5.0";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::ConstantStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_constant(pair);
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
        let spec = "input a: Int, b: Int, c: Bool";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::InputStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let inputs = parser.parse_inputs(pair);
        assert_eq!(inputs.len(), 3);
        assert_eq!(format!("{}", inputs[0]), "input a: Int");
        assert_eq!(format!("{}", inputs[1]), "input b: Int");
        assert_eq!(format!("{}", inputs[2]), "input c: Bool");
    }

    #[test]
    fn parse_mirror_ast() {
        let spec = "output a mirrors b when 3 > 5";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::MirrorStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mirror = parser.parse_mirror(pair).expect("failed to parse eval condition");
        assert_eq!(format!("{}", mirror), "output a mirror b when 3 > 5");
    }

    #[test]
    fn build_ast_parameterized_input() {
        let spec = "input in (ab: Int8): Int8\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_output() {
        parses_to! {
            parser: LolaParser,
            input:  "output out: Int := in + 1",
            rule:   Rule::OutputStream,
            tokens: [
                OutputStream(0, 25, [
                    NamedOutputDecl(0, 10, [Ident(7, 10)]),
                    Type(12, 15, [
                        Ident(12, 15, []),
                    ]),
                    SimpleEvalDecl(16, 25, [
                        Expr(19, 25, [
                            Ident(19, 21, []),
                            Add(22, 23, []),
                            Literal(24, 25, [
                                NumberLiteral(24, 25, [
                                    NumberLiteralValue(24, 25, [])
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_output_ast() {
        let spec = "output out: Int eval with in + 1";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::OutputStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_output(pair).unwrap();
        assert_eq!(format!("{}", ast), spec)
    }

    #[test]
    fn parse_trigger() {
        parses_to! {
            parser: LolaParser,
            input:  "trigger in != out \"some message\"",
            rule:   Rule::SimpleTrigger,
            tokens: [
                SimpleTrigger(0, 32, [
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
        let spec = "trigger in ≠ out \"some message\"";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::SimpleTrigger, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_trigger(pair).unwrap();
        assert_eq!(format!("{}", ast), "trigger eval when in ≠ out with \"some message\"")
    }

    #[test]
    fn parse_complex_trigger() {
        let spec =
            "trigger (p) spawn when a = 0 with b eval @1Hz when b = 0 with \"msg\".format(a) close @2Hz when true";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::OutputStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_output(pair).unwrap();
        assert_eq!(format!("{}", ast), spec)
    }

    #[test]
    fn trigger_missing_message() {
        let spec = "trigger eval when a = 0";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::OutputStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_output(pair).unwrap();
        assert_eq!(format!("{}", ast), "trigger eval when a = 0 with \"\"");
    }

    #[test]
    fn trigger_missing_message2() {
        let spec = "trigger a = 0";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        let pair = LolaParser::parse(Rule::SimpleTrigger, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_trigger(pair).unwrap();
        assert_eq!(format!("{}", ast), "trigger eval when a = 0 with \"\"");
    }

    #[test]
    fn parse_expression() {
        let content = "in + 1";
        let config = ParserConfig::for_string(content.into());
        let parser = RtLolaParser::new(&config);
        let expr = LolaParser::parse(Rule::Expr, content)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.build_expression_ast(expr.into_inner()).unwrap();
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_expression_precedence() {
        let content = "(a ∨ b ∧ c)";
        let config = ParserConfig::for_string(content.into());
        let parser = RtLolaParser::new(&config);
        let expr = LolaParser::parse(Rule::Expr, content)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.build_expression_ast(expr.into_inner()).unwrap();
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_missing_closing_parenthesis() {
        let content = "(a ∨ b ∧ c";
        let config = ParserConfig::for_string(content.into());
        let parser = RtLolaParser::new(&config);
        let expr = LolaParser::parse(Rule::Expr, content)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.build_expression_ast(expr.into_inner()).unwrap();
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn build_simple_ast() {
        let spec = "input in: Int\noutput out: Int eval with in\ntrigger eval when in ≠ out with \"\"\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ast_input() {
        let spec = "input in: Int\ninput in2: Int\ninput in3: (Int, Bool)\ninput in4: Bool\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parenthesized_expression() {
        let spec = "output s: Bool eval with (true ∨ true)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_optional_type() {
        let spec = "output s: Bool? eval with (false ∨ true)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_default() {
        let spec = "output s: Int eval with s.offset(by: -1).defaults(to: (3 * 4))\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_hold() {
        let spec = "output s: Int eval with s.offset(by: -1).hold().defaults(to: 3 * 4)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_get_expression_def() {
        let spec = "output s: Int eval with s.get().defaults(to: -1)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_check_expression_def() {
        let spec = "input i: Int\noutput s: Int eval @1Hz with if i.is_fresh() then -1 else 1\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ternary_expression() {
        let spec = "input in: Int\noutput s: Int eval with if in = 3 then 4 else in + 2\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_function_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int eval with nroot(1, sin(1, in))\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger() {
        let spec = "input in: Int\ntrigger eval when in > 5 with \"\"\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger_extend() {
        let spec = "trigger eval @1Hz when in > 5 with \"\"\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger_message() {
        let spec = "trigger eval when in > 5 with \"test trigger\"\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
        let eval = &ast.outputs[0].eval[0];
        assert!(
            matches!(&eval.eval_expression, Some(Expression {kind: ExpressionKind::Lit(Literal {kind:LitKind::Str(s),..}),..}) if s == "test trigger")
        );
    }

    #[test]
    fn build_simple_trigger() {
        let spec = "trigger x > 10 \"msg\"";
        let reference = "trigger eval when x > 10 with \"msg\"\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, reference);
    }

    #[test]
    fn build_complex_expression() {
        let spec =
            "output s: Double eval with if !((s.offset(by: -1).defaults(to: (3 * 4)) + -4) = 12) ∨ true = false then 2.0 else 4.1\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_type_declaration() {
        let spec = "type VerifiedUser { name: String }\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parameter_list() {
        let spec = "output s (a: B, c: D): E eval with 3\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_termination_spec() {
        let spec = "output s (a: Int): Int eval with 3 close when s > 10\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_tuple_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int eval with (1, in.0).1\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_string() {
        let spec = r#"constant s: String := "a string with \n newline"
"#;
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_raw_string() {
        let spec = r##"constant s: String := r#"a raw \ string that " needs padding"#
"##;
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_import() {
        let spec = "import math\ninput in: UInt8\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_max() {
        let spec = "import math\ninput a: Int32\ninput b: Int32\noutput maxres: Int32 eval with max<Int32>(a, b)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call() {
        let spec = "output count eval with count.offset(-1).default(0) + 1\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call_with_param() {
        let spec = "output count eval with count.offset<Int8>(-1).default(0) + 1\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_realtime_offset() {
        let spec = "output a eval with b.offset(by: -1s)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_future_offset() {
        let spec = "output a eval with b.offset(by: 1)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_function_argument_name() {
        let spec = "output a eval with b.hold().defaults(to: 0)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_precedence_not_regression() {
        parses_to! {
            parser: LolaParser,
            input:  "!(fast || false) && fast",
            rule:   Rule::Expr,
            tokens: [
                Expr(0, 24, [
                    UnaryExpr(0, 16, [
                        Neg(0, 1, []),
                        ParenthesizedExpression(1, 16, [
                            OpeningParenthesis(1, 2, []),
                            Expr(2, 15, [
                                Ident(2, 6, []),
                                Or(7, 9, []),
                                Literal(10, 15, [
                                    False(10, 15, [])
                                ])
                            ]),
                            ClosingParenthesis(15, 16, [])
                        ])
                    ]),
                    And(17, 19, []),
                    Ident(20, 24, [])
                ]),
            ]
        };
    }

    #[test]
    fn handle_bom() {
        let spec = "\u{feff}input a: Bool\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "input a: Bool\n");
    }

    #[test]
    fn regression71_and_simple_decl() {
        let spec = "output outputstream := 42 output c := outputstream";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output outputstream eval with 42\noutput c eval with outputstream\n",
        );
    }

    #[test]
    fn parse_bitwise() {
        let spec = "output x eval with 1 ^ 0 & 23123 | 111\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn spawn_no_expression() {
        let spec = "output x spawn when true eval with 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn spawn_no_cond_with_pacing() {
        let spec = "output x (y: Int32) spawn @1Hz with 1 eval with 5 close @1Hz when true\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_template_spec() {
        let spec = "output x (y: Int8) spawn with 42 eval @1Hz when y = 1337 with 5 close when false\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_full_template_spec_type() {
        let spec =
            "output x (y: Int8): Bool spawn when y = 3 with 42 eval @1Hz when y = 1337 with 5 close when false\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_full_template_spec() {
        //spawn eval close
        let spec = "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5 close when false\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5 close when false\n",
        );
    }

    #[test]
    fn build_full_order1() {
        // close eval spawn
        let spec = "output x (y: Int8) close when false eval @1Hz when y = 1337 with 5 spawn when y = 17 with 42\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5 close when false\n",
        );
    }

    #[test]
    fn build_full_order2() {
        // eval close spawn
        let spec = "output x (y: Int8) eval @1Hz when y = 1337 with 5  close when false spawn when y = 17 with 42\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5 close when false\n",
        );
    }

    #[test]
    fn build_full_order3() {
        // eval spawn close
        let spec = "output x (y: Int8) eval @1Hz when y = 1337 with 5 spawn when y = 17 with 42 close when false\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5 close when false\n",
        );
    }

    #[test]
    fn build_full_order4() {
        // close spawn eval
        let spec = "output x (y: Int8) close when false spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5 close when false\n",
        );
    }

    #[test]
    fn build_full_order5() {
        // spawn close eval
        let spec = "output x (y: Int8) spawn when y = 17 with 42 close when false eval @1Hz when y = 1337 with 5\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1337 with 5 close when false\n",
        );
    }

    #[test]
    fn multiple_eval_inputs() {
        // eval eval eval
        let spec = "output x (y: Int8) eval @1Hz when y = 1336 with 4 eval @1Hz when y = 1337 with 5 eval @1Hz when y = 1338 with 6\n";
        let ast = parse_without_desugar(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn multiple_eval_inputs_with_spawn() {
        // eval eval eval spawn
        let spec = "output x (y: Int8) eval @1Hz when y = 1336 with 4 eval @1Hz when y = 1337 with 5 eval @1Hz when y = 1338 with 6 spawn when y = 17 with 42\n";
        let ast = parse_without_desugar(spec);
        cmp_ast_spec(&ast, "output x (y: Int8) spawn when y = 17 with 42 eval @1Hz when y = 1336 with 4 eval @1Hz when y = 1337 with 5 eval @1Hz when y = 1338 with 6\n");
    }

    #[test]
    fn multiple_eval_inputs_with_close() {
        // close eval eval eval
        let spec = "output x (y: Int8) close when false eval when y = 1336 with 4 eval when y = 1337 with 5 eval when y = 1338 with 6\n";
        let ast = parse_without_desugar(spec);
        cmp_ast_spec(&ast, "output x (y: Int8) eval when y = 1336 with 4 eval when y = 1337 with 5 eval when y = 1338 with 6 close when false\n");
        //let ast2 = parse(spec);
        //cmp_ast_spec(&ast2, "output x (y: Int8) eval when y = 1336 ∨ y = 1337 ∨ y = 1338 with if y = 1336 then 4 else if y = 1337 then 5 else 6 close when false\n");
    }

    #[test]
    fn build_full_template_spec_type_two_eval() {
        // spawn eval eval close
        let spec = "output x (y: Int8): Bool spawn when y = 17 with 42 eval @1Hz when y = 1336 with 4 eval when y = 1337 with 5 close when false\n";
        let ast = parse_without_desugar(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_multiple_close() {
        let spec = "output x close when true eval with 5 close when false\n";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        match parser.parse_spec() {
            Ok(_) => panic!("Expected error"),
            Err(e) => assert_eq!(e.num_errors(), 1),
        }
    }

    #[test]
    fn spawn_parts_order1() {
        let spec = "output x (y: Int8) spawn with 42 when y = 1 eval with 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "output x (y: Int8) spawn when y = 1 with 42 eval with 5\n");
    }

    #[test]
    fn spawn_parts_order2() {
        let spec = "output x (y: Int8) spawn when y = 1 with 42 eval with 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn eval_parts_order1() {
        let spec = "output x (y: Int8) spawn with 42 eval with 5 when y = 42\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "output x (y: Int8) spawn with 42 eval when y = 42 with 5\n");
    }

    #[test]
    fn eval_parts_order2() {
        let spec = "output x (y: Int8) spawn with 42 eval when y = 42 with 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn spawn_duplicate_when() {
        let spec = "output x spawn when true when true eval with 5\n";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);

        match parser.parse_spec() {
            Ok(_) => panic!("Expected error"),
            Err(e) => {
                //use rtlola_reporting::Handler;
                //let h = Handler::from(ParserConfig::for_string(spec.to_string()));
                //h.emit_error(&e);
                assert_eq!(e.num_errors(), 1)
            },
        }
    }

    #[test]
    fn spawn_duplicate_with() {
        let spec = "output x (p) spawn @1Hz with 3 with 3 eval with 5\n";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        match parser.parse_spec() {
            Ok(_) => panic!("Expected error"),
            Err(e) => {
                //use rtlola_reporting::Handler;
                //let h = Handler::from(ParserConfig::for_string(spec.to_string()));
                //h.emit_error(&e);
                assert_eq!(e.num_errors(), 1)
            },
        }
    }

    #[test]
    fn duplicate_close_clauses() {
        let spec = "output x eval with 5 close when true close when x == 5\n";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        match parser.parse_spec() {
            Ok(_) => panic!("Expected error"),
            Err(e) => {
                //use rtlola_reporting::Handler;
                //let h = Handler::from(ParserConfig::for_string(spec.to_string()));
                //h.emit_error(&e);
                assert_eq!(e.num_errors(), 1)
            },
        }
    }

    #[test]
    fn spawn_no_expr_no_condition_np_pacing() {
        let spec = "output x spawn eval with 5\n";
        let config = ParserConfig::for_string(spec.into());
        let parser = RtLolaParser::new(&config);
        match parser.parse_spec() {
            Ok(_) => panic!("Expected error"),
            Err(e) => assert_eq!(e.num_errors(), 1),
        }
    }

    #[test]
    fn eval_no_expr() {
        let spec = "output x (y: Int8) spawn when true with 42 eval @1Hz when y = 42\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "output x (y: Int8) spawn when true with 42 eval @1Hz when y = 42 with ()\n",
        );
    }

    #[test]
    fn valid_percentile() {
        let spec = "output x eval with x.aggregate(over: 3s, using: pctl15)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn spawn_with_pacing() {
        let spec = "output x eval with 5 spawn @3Hz with (x)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "output x spawn @3Hz with (x) eval with 5\n");
    }

    #[test]
    fn test_instance_window() {
        let spec = "input a: Int32\n\
        output b (p: Bool) spawn with a = 42 eval with a\n\
        output c eval @1Hz with b(false).aggregate(over: 1s, using: Σ)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_impl_simpl() {
        let spec = "input a: Bool\n\
        input b: Bool\n\
        output c eval with a -> b\n";
        let ast = parse(spec);
        cmp_ast_spec(
            &ast,
            "input a: Bool\n\
            input b: Bool\n\
            output c eval with !a ∨ b\n",
        );
    }

    #[test]
    fn parse_impl_right_associative() {
        let spec = "input a: Bool\ninput b: Bool\noutput c eval with a -> b -> c\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "input a: Bool\ninput b: Bool\noutput c eval with !a ∨ !b ∨ c\n");
    }

    #[test]
    fn parse_impl_left() {
        let spec = "input a: Bool\ninput b: Bool\noutput c eval with (a -> b) -> c\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "input a: Bool\ninput b: Bool\noutput c eval with !(!a ∨ b) ∨ c\n");
    }

    #[test]
    fn parse_impl_nested() {
        let spec = "input a: Bool\ninput b: Bool\noutput c eval with a ∧ b -> c\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "input a: Bool\ninput b: Bool\noutput c eval with !(a ∧ b) ∨ c\n");
    }

    #[test]
    fn instance_aggregation_simpl_fresh() {
        let spec = "input a: Int32\n\
        output b (p) spawn with a eval when a > 5 with b(p).offset(by: -1).defaults(to: 0) + 1\n\
        output c eval with b.aggregate(over_instances: fresh, using: Σ)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }
    #[test]
    fn instance_aggregation_simpl_all() {
        let spec = "input a: Int32\n\
        output b (p) spawn with a eval when a > 5 with b(p).offset(by: -1).defaults(to: 0) + 1\n\
        output c eval with b.aggregate(over_instances: all, using: Σ)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }
}
