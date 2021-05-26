//! This module contains the parser for the Lola Language.

use std::cell::RefCell;
use std::rc::Rc;
use std::str::FromStr;

use lazy_static::lazy_static;
use num::rational::Rational64 as Rational;
use num::traits::Pow;
use num::{BigInt, BigRational, FromPrimitive, ToPrimitive};
use pest::iterators::{Pair, Pairs};
use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;
use pest_derive::Parser;
use rtlola_reporting::{Handler, Span};

use super::ast::*;
use crate::ast::Literal;
use crate::ParserConfig;

#[derive(Parser)]
#[grammar = "lola.pest"]
struct LolaParser;

#[derive(Debug, Clone)]
pub(crate) struct RtLolaParser<'a> {
    spec: RtLolaAst,
    handler: &'a Handler,
    config: ParserConfig,
    node_id: RefCell<NodeId>,
}

lazy_static! {
    // precedence taken from C/C++: https://en.wikipedia.org/wiki/Operators_in_C_and_C++
    // Precedence climber can be used to build the AST, see https://pest-parser.github.io/book/ for more details
    static ref PREC_CLIMBER: PrecClimber<Rule> = {
        use self::Assoc::*;
        use self::Rule::*;

        PrecClimber::new(vec![
            Operator::new(Or, Left),
            Operator::new(And, Left),
            Operator::new(BitOr, Left),
            Operator::new(BitXor, Left),
            Operator::new(BitAnd, Left),
            Operator::new(Equal, Left) | Operator::new(NotEqual, Left),
            Operator::new(LessThan, Left) | Operator::new(LessThanOrEqual, Left) | Operator::new(MoreThan, Left) | Operator::new(MoreThanOrEqual, Left),
            Operator::new(ShiftLeft, Left) | Operator::new(ShiftRight, Left),
            Operator::new(Add, Left) | Operator::new(Subtract, Left),
            Operator::new(Multiply, Left) | Operator::new(Divide, Left) | Operator::new(Mod, Left),
            Operator::new(Power, Right),
            Operator::new(Dot, Left),
            Operator::new(OpeningBracket, Left),
        ])
    };
}

impl<'a> RtLolaParser<'a> {
    pub(crate) fn new(handler: &'a Handler, config: ParserConfig) -> Self {
        RtLolaParser {
            spec: RtLolaAst::empty(),
            handler,
            config,
            node_id: RefCell::new(NodeId::new(0)),
        }
    }

    /// Transforms a textual representation of a Lola specification into
    /// an AST representation.
    pub(crate) fn parse(handler: &'_ Handler, config: ParserConfig) -> Result<RtLolaAst, pest::error::Error<Rule>> {
        RtLolaParser::new(handler, config).parse_spec()
    }

    /// Runs the parser on the give spec.
    pub(crate) fn parse_spec(mut self) -> Result<RtLolaAst, pest::error::Error<Rule>> {
        let mut pairs = LolaParser::parse(Rule::Spec, &self.config.spec)?;
        assert!(pairs.clone().count() == 1, "Spec must not be empty.");
        let spec_pair = pairs.next().unwrap();
        assert!(spec_pair.as_rule() == Rule::Spec);
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
                Rule::OutputStream => {
                    let output = self.parse_output(pair);
                    self.spec.outputs.push(Rc::new(output));
                },
                Rule::Trigger => {
                    let trigger = self.parse_trigger(pair);
                    self.spec.trigger.push(Rc::new(trigger));
                },
                Rule::TypeDecl => {
                    let type_decl = self.parse_type_declaration(pair);
                    self.spec.type_declarations.push(type_decl);
                },
                Rule::EOI => {},
                _ => unreachable!(),
            }
        }
        Ok(self.spec)
    }

    fn next_id(&self) -> NodeId {
        let res = *self.node_id.borrow();
        self.node_id.borrow_mut().0 += 1;
        res
    }

    fn parse_import(&self, pair: Pair<Rule>) -> Import {
        assert_eq!(pair.as_rule(), Rule::ImportStmt);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        Import {
            name,
            id: self.next_id(),
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
            id: self.next_id(),
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
                id: self.next_id(),
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
     * Transforms a `Rule::OutputStream` into `Output` AST node.
     * Panics if input is not `Rule::OutputStream`.
     * The output rule consists of the following tokens:
     * - `Rule::Ident`
     * - `Rule::Type`
     * - `Rule::Expr`
     */
    fn parse_output(&self, pair: Pair<'_, Rule>) -> Output {
        assert_eq!(pair.as_rule(), Rule::OutputStream);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));

        let mut pair = pairs.next().expect("mismatch between grammar and AST");
        let params = if let Rule::ParamList = pair.as_rule() {
            let res = self.parse_parameter_list(pair.into_inner());
            pair = pairs.next().expect("mismatch between grammar and AST");

            res
        } else {
            Vec::new()
        };

        let ty = if let Rule::Type = pair.as_rule() {
            let ty = self.parse_type(pair);
            pair = pairs.next().expect("mismatch between grammar and AST");
            Some(ty)
        } else {
            None
        };

        // Parse the `@ [Expr]` part of output declaration
        let extend = if let Rule::ActivationCondition = pair.as_rule() {
            let span: Span = pair.as_span().into();
            let expr = self.build_expression_ast(pair.into_inner());
            pair = pairs.next().expect("mismatch between grammar and AST");
            ActivationCondition {
                expr: Some(expr),
                id: self.next_id(),
                span,
            }
        } else {
            ActivationCondition {
                expr: None,
                id: self.next_id(),
                span: Span::Unknown,
            }
        };

        let spawn = if let Rule::SpawnDecl = pair.as_rule() {
            let spawn_spec = self.parse_spawn_spec(pair);
            pair = pairs.next().expect("mismatch between grammar and AST");
            Some(spawn_spec)
        } else {
            None
        };

        let filter = if let Rule::FilterDecl = pair.as_rule() {
            let filter_spec = self.parse_filter_spec(pair);
            pair = pairs.next().expect("mismatch between grammar and AST");
            Some(filter_spec)
        } else {
            None
        };

        let close = if let Rule::CloseDecl = pair.as_rule() {
            let close_spec = self.parse_close_spec(pair);
            pair = pairs.next().expect("mismatch between grammar and AST");
            Some(close_spec)
        } else {
            None
        };

        // Parse expression
        let expression = self.build_expression_ast(pair.into_inner());
        Output {
            id: self.next_id(),
            name,
            ty,
            extend,
            params: params.into_iter().map(Rc::new).collect(),
            spawn,
            filter,
            close,
            expression,
            span,
        }
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
                id: self.next_id(),
                span,
            });
        }
        params
    }

    fn parse_spawn_spec(&self, spawn_pair: Pair<'_, Rule>) -> SpawnSpec {
        let span_inv: Span = spawn_pair.as_span().into();

        let mut spawn_children = spawn_pair.into_inner();
        let mut next_pair = spawn_children.next();

        let pacing = if let Some(pair) = next_pair.clone() {
            if let Rule::ActivationCondition = pair.as_rule() {
                let span: Span = pair.as_span().into();
                let expr = self.build_expression_ast(pair.into_inner());
                next_pair = spawn_children.next();
                ActivationCondition {
                    expr: Some(expr),
                    id: self.next_id(),
                    span,
                }
            } else {
                ActivationCondition {
                    expr: None,
                    id: self.next_id(),
                    span: Span::Unknown,
                }
            }
        } else {
            ActivationCondition {
                expr: None,
                id: self.next_id(),
                span: Span::Unknown,
            }
        };

        let target = if let Some(pair) = next_pair.clone() {
            if let Rule::SpawnWith = pair.as_rule() {
                let target_pair = pair.into_inner().next().expect("mismatch between grammar and AST");
                let target_exp = self.build_expression_ast(target_pair.into_inner());
                next_pair = spawn_children.next();
                Some(target_exp)
            } else {
                None
            }
        } else {
            None
        };

        let (condition, is_if) = if let Some(pair) = next_pair {
            let is_if = match pair.as_rule() {
                Rule::SpawnIf => true,
                Rule::SpawnUnless => false,
                _ => unreachable!(),
            };
            let condition_pair = pair.into_inner().next().expect("mismatch between grammar and AST");
            let condition_exp = self.build_expression_ast(condition_pair.into_inner());
            (Some(condition_exp), is_if)
        } else {
            (None, false)
        };

        if target.is_none() && condition.is_none() && pacing.expr.is_none() {
            self.handler.error_with_span(
                "Spawn condition needs either expression or condition",
                span_inv.clone(),
                Some("found spawn condition here"),
            );
        }
        SpawnSpec {
            target,
            pacing,
            condition,
            is_if,
            id: self.next_id(),
            span: span_inv,
        }
    }

    fn parse_filter_spec(&self, ext_pair: Pair<'_, Rule>) -> FilterSpec {
        let span_ext: Span = ext_pair.as_span().into();

        let mut children = ext_pair.into_inner();

        let first_child = children.next().expect("mismatch between grammar and ast");
        let target = match first_child.as_rule() {
            Rule::Expr => self.build_expression_ast(first_child.into_inner()),
            _ => unreachable!(),
        };
        FilterSpec {
            target,
            id: self.next_id(),
            span: span_ext,
        }
    }

    fn parse_close_spec(&self, close_pair: Pair<'_, Rule>) -> CloseSpec {
        let span_close: Span = close_pair.as_span().into();

        let mut children = close_pair.into_inner();

        let first_child = children.next().expect("mismatch between grammar and ast");
        let target = match first_child.as_rule() {
            Rule::Expr => self.build_expression_ast(first_child.into_inner()),
            _ => unreachable!(),
        };
        CloseSpec {
            target,
            id: self.next_id(),
            span: span_close,
        }
    }

    /**
     * Transforms a `Rule::Trigger` into `Trigger` AST node.
     * Panics if input is not `Rule::Trigger`.
     * The output rule consists of the following tokens:
     * - (`Rule::Ident`)?
     * - `Rule::Expr`
     * - (`Rule::StringLiteral`)?
     */
    fn parse_trigger(&self, pair: Pair<'_, Rule>) -> Trigger {
        assert_eq!(pair.as_rule(), Rule::Trigger);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();

        let mut pair = pairs.next().expect("mismatch between grammar and AST");

        // Parse the `@ [Expr]` part of output declaration
        let extend = if let Rule::ActivationCondition = pair.as_rule() {
            let span: Span = pair.as_span().into();
            let expr = self.build_expression_ast(pair.into_inner());
            pair = pairs.next().expect("mismatch between grammar and AST");
            ActivationCondition {
                expr: Some(expr),
                id: self.next_id(),
                span,
            }
        } else {
            ActivationCondition {
                expr: None,
                id: self.next_id(),
                span: Span::Unknown,
            }
        };

        let expression = self.build_expression_ast(pair.into_inner());

        let message = pairs.next().map(|pair| {
            assert_eq!(pair.as_rule(), Rule::String);
            pair.as_str().to_string()
        });

        // Parse list of info streams
        let info_streams = pairs
            .next()
            .map(|pair| {
                assert_eq!(pair.as_rule(), Rule::IdentList);
                pair.into_inner().into_iter().map(|p| self.parse_ident(&p)).collect()
            })
            .unwrap_or_default();

        Trigger {
            id: self.next_id(),
            expression,
            extend,
            message,
            info_streams,
            span,
        }
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
                id: self.next_id(),
                span: pair.as_span().into(),
            }));
        }

        TypeDeclaration {
            name: Some(name),
            span,
            id: self.next_id(),
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
                    return Type::new_simple(self.next_id(), pair.as_str().to_string(), pair.as_span().into());
                },
                Rule::Type => tuple.push(self.parse_type(pair)),
                Rule::Optional => {
                    let span = pair.as_span();
                    let inner = pair
                        .into_inner()
                        .next()
                        .expect("mismatch between grammar and AST: first argument is a type");
                    let inner_ty = Type::new_simple(self.next_id(), inner.as_str().to_string(), inner.as_span().into());
                    return Type::new_optional(self.next_id(), inner_ty, span.into());
                },
                _ => unreachable!("{:?} is not a type, ensured by grammar", pair.as_rule()),
            }
        }
        Type::new_tuple(self.next_id(), tuple, span.into())
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
                Literal::new_str(self.next_id(), str_rep, inner.as_span().into())
            },
            Rule::RawString => {
                let str_rep = inner.as_str();
                Literal::new_raw_str(self.next_id(), str_rep, inner.as_span().into())
            },
            Rule::NumberLiteral => {
                let span = inner.as_span();
                let mut pairs = inner.into_inner();
                let value = pairs.next().expect("Mismatch between AST and grammar");

                let str_rep: &str = value.as_str();
                let unit = pairs.next().map(|unit| unit.as_str().to_string());

                Literal::new_numeric(self.next_id(), str_rep, unit, span.into())
            },
            Rule::True => Literal::new_bool(self.next_id(), true, inner.as_span().into()),
            Rule::False => Literal::new_bool(self.next_id(), false, inner.as_span().into()),
            _ => unreachable!(),
        }
    }

    #[allow(clippy::vec_box)]
    fn parse_vec_of_expressions(&self, pairs: Pairs<'_, Rule>) -> Vec<Expression> {
        pairs.map(|expr| self.build_expression_ast(expr.into_inner())).collect()
    }

    fn parse_vec_of_types(&self, pairs: Pairs<'_, Rule>) -> Vec<Type> {
        pairs.map(|p| self.parse_type(p)).collect()
    }

    fn build_function_expression(&self, pair: Pair<'_, Rule>, span: Span) -> Expression {
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
            args.push(self.build_expression_ast(pair.into_inner()));
        }
        let name = FunctionName {
            name: fun_name,
            arg_names,
        };
        Expression::new(self.next_id(), ExpressionKind::Function(name, type_params, args), span)
    }

    /**
     * Builds the Expr AST.
     */
    fn build_expression_ast(&self, pairs: Pairs<'_, Rule>) -> Expression {
        PREC_CLIMBER.climb(
            pairs,
            |pair: Pair<'_, Rule>| self.build_term_ast(pair),
            |lhs: Expression, op: Pair<'_, Rule>, rhs: Expression| {
                // Reduce function combining `Expression`s to `Expression`s with the correct precs
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
                            _ => (None, span.clone(), Box::new(lhs)),
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
                                        self.handler.error_with_span(
                                            &format!("expected unsigned integer, found {}", l),
                                            rhs.span,
                                            Some("unexpected"),
                                        );
                                        std::process::exit(1);
                                    }
                                };
                                let binop_expr =
                                    Expression::new(self.next_id(), ExpressionKind::Field(inner, ident), binop_span);
                                match unop {
                                    None => return binop_expr,
                                    Some(unop) => {
                                        return Expression::new(
                                            self.next_id(),
                                            ExpressionKind::Unary(unop, Box::new(binop_expr)),
                                            span,
                                        )
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
                                        let offset = match offset_expr.parse_offset() {
                                            Ok(offset) => offset,
                                            Err(reason) => {
                                                self.handler.error_with_span(
                                                    "failed to parse offset",
                                                    rhs.span,
                                                    Some(&reason),
                                                );
                                                std::process::exit(1);
                                            }
                                        };
                                        ExpressionKind::Offset(inner, offset)
                                    }
                                    "hold()" => {
                                        assert_eq!(args.len(), 0);
                                        ExpressionKind::StreamAccess(inner, StreamAccessKind::Hold)
                                    }
                                    "hold(or:)" => {
                                        assert_eq!(args.len(), 1);
                                        let lhs = Expression::new(
                                            self.next_id(),
                                            ExpressionKind::StreamAccess(inner, StreamAccessKind::Hold),
                                            span.clone(),
                                        );
                                        ExpressionKind::Default(Box::new(lhs), Box::new(args[0].clone()))
                                    }
                                    "get()" => {
                                        assert_eq!(args.len(), 0);
                                        ExpressionKind::StreamAccess(inner, StreamAccessKind::Optional)
                                    }
                                    "aggregate(over_discrete:using:)" | "aggregate(over_exactly_discrete:using:)" |"aggregate(over:using:)" | "aggregate(over_exactly:using:)" => {
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
                                                fun => {
                                                    self.handler.error_with_span(
                                                        &format!("unknown aggregation function {}", fun),
                                                        i.span.clone(),
                                              Some("available: count, min, max, sum, average, exists, forall, integral"),
                                                    );
                                                    std::process::exit(1);
                                                }
                                            },
                                            _ => {
                                                self.handler.error_with_span(
                                                    "expected aggregation function",
                                                    args[1].span.clone(),
                                                    Some("available: count, min, max, sum, average, integral"),
                                                );
                                                std::process::exit(1);
                                            }
                                        };
                                        if signature.contains("discrete") {
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
                                let binop_expr = Expression::new(self.next_id(), kind, binop_span);
                                match unop {
                                    None => return binop_expr,
                                    Some(unop) => {
                                        return Expression::new(
                                            self.next_id(),
                                            ExpressionKind::Unary(unop, Box::new(binop_expr)),
                                            span,
                                        )
                                    }
                                }
                            }
                            _ => {
                                self.handler.error_with_span(
                                    &format!("expected method call or tuple access, found {}", rhs),
                                    rhs.span,
                                    Some("unexpected"),
                                );
                                std::process::exit(1);
                            }
                        }
                    }
                    Rule::OpeningBracket => {
                        let offset = match rhs.parse_offset() {
                            Ok(offset) => offset,
                            Err(reason) => {
                                self.handler.error_with_span(
                                    "failed to parse offset expression",
                                    rhs.span,
                                    Some(&reason),
                                );
                                std::process::exit(1);
                            }
                        };
                        match lhs.kind {
                            ExpressionKind::Unary(unop, inner) => {
                                let inner_span = inner.span.union(&rhs.span);
                                let new_inner =
                                    Expression::new(self.next_id(), ExpressionKind::Offset(inner, offset), inner_span);
                                return Expression::new(
                                    self.next_id(),
                                    ExpressionKind::Unary(unop, Box::new(new_inner)),
                                    span,
                                );
                            }
                            _ => {
                                return Expression::new(
                                    self.next_id(),
                                    ExpressionKind::Offset(lhs.into(), offset),
                                    span,
                                )
                            }
                        }
                    }
                    _ => unreachable!(),
                };
                Expression::new(self.next_id(), ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)), span)
            },
        )
    }

    fn build_term_ast(&self, pair: Pair<'_, Rule>) -> Expression {
        let span = pair.as_span();
        match pair.as_rule() {
            // Map function from `Pair` to AST data structure `Expression`
            Rule::Literal => {
                Expression::new(
                    self.next_id(),
                    ExpressionKind::Lit(self.parse_literal(pair)),
                    span.into(),
                )
            },
            Rule::Ident => {
                Expression::new(
                    self.next_id(),
                    ExpressionKind::Ident(self.parse_ident(&pair)),
                    span.into(),
                )
            },
            Rule::ParenthesizedExpression => {
                let mut inner = pair.into_inner();
                let opp = inner.next().expect(
                    "Rule::ParenthesizedExpression has a token for the (potentialy missing) opening parenthesis",
                );
                let opening_parenthesis = if let Rule::OpeningParenthesis = opp.as_rule() {
                    Some(Box::new(Parenthesis::new(self.next_id(), opp.as_span().into())))
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
                    Some(Box::new(Parenthesis::new(self.next_id(), closing.as_span().into())))
                } else {
                    None
                };

                Expression::new(
                    self.next_id(),
                    ExpressionKind::ParenthesizedExpression(
                        opening_parenthesis,
                        Box::new(self.build_expression_ast(inner_expression.into_inner())),
                        closing_parenthesis,
                    ),
                    span.into(),
                )
            },
            Rule::UnaryExpr => {
                // First child is the operator, second the operand.
                let mut children = pair.into_inner();
                let pest_operator = children.next().expect("Unary expressions need to have an operator.");
                let operand = children.next().expect("Unary expressions need to have an operand.");
                let operand = self.build_term_ast(operand);
                let operator = match pest_operator.as_rule() {
                    Rule::Add => return operand, // Discard unary plus because it is semantically null.
                    Rule::Subtract => UnOp::Neg,
                    Rule::Neg => UnOp::Not,
                    Rule::BitNot => UnOp::BitNot,
                    _ => unreachable!(),
                };
                Expression::new(
                    self.next_id(),
                    ExpressionKind::Unary(operator, Box::new(operand)),
                    span.into(),
                )
            },
            Rule::TernaryExpr => {
                let mut children = self.parse_vec_of_expressions(pair.into_inner());
                assert_eq!(children.len(), 3, "A ternary expression needs exactly three children.");
                Expression::new(
                    self.next_id(),
                    ExpressionKind::Ite(
                        Box::new(children.remove(0)),
                        Box::new(children.remove(0)),
                        Box::new(children.remove(0)),
                    ),
                    span.into(),
                )
            },
            Rule::Tuple => {
                let elements = self.parse_vec_of_expressions(pair.into_inner());
                assert!(elements.len() != 1, "Tuples may not have exactly one element.");
                Expression::new(self.next_id(), ExpressionKind::Tuple(elements), span.into())
            },
            Rule::Expr => self.build_expression_ast(pair.into_inner()),
            Rule::FunctionExpr => self.build_function_expression(pair, span.into()),
            Rule::IntegerLiteral => {
                let span: Span = span.clone().into();
                Expression::new(
                    self.next_id(),
                    ExpressionKind::Lit(Literal::new_numeric(self.next_id(), pair.as_str(), None, span.clone())),
                    span,
                )
            },
            Rule::MissingExpression => {
                let span = span.into();
                Expression::new(self.next_id(), ExpressionKind::MissingExpression, span)
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
            Err(e) => return Err(format!("parsing rational '{}' failed: {}", repr, e)),
        };
        let mut r = BigRational::from(integer);
        if !dec_digits.is_empty() {
            // divide by 10 for each decimal place
            r /= BigInt::from_u8(10).unwrap().pow(dec_digits.len());
        }

        if !exp_str.is_empty() {
            let exp = match BigInt::from_str(exp_str) {
                Ok(i) => i,
                Err(e) => return Err(format!("parsing rational '{}' failed: {}", repr, e)),
            };
            let exp = match exp.to_i16() {
                Some(i) => i,
                None => {
                    return Err(format!(
                        "parsing rational '{}' failed: e exponent {} does not fit into i16",
                        repr, exp
                    ))
                },
            };
            let factor = BigInt::from_u8(10).unwrap().pow(exp.abs() as u16);
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
                    "parsing rational failed: rational {} does not fit into Rational64",
                    r
                ))
            },
        };
        Ok(Rational::from(p))
    }
}

#[cfg(test)]
mod tests {

    use std::path::PathBuf;

    use pest::{consumes_to, parses_to};

    use super::*;

    fn create_parser<'a>(handler: &'a Handler, spec: &'a str) -> RtLolaParser<'a> {
        RtLolaParser::new(handler, ParserConfig::for_string(spec.into()))
    }

    fn parse(spec: &str) -> RtLolaAst {
        super::super::parse(ParserConfig::for_string(spec.into())).unwrap_or_else(|e| panic!("{}", e))
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
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), spec.into());
        let parser = create_parser(&handler, spec);
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
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), spec.into());
        let parser = create_parser(&handler, spec);
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
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), spec.into());
        let parser = create_parser(&handler, spec);
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
                    Ident(7, 10, []),
                    Type(12, 15, [
                        Ident(12, 15, []),
                    ]),
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
            ]
        };
    }

    #[test]
    fn parse_output_ast() {
        let spec = "output out: Int := in + 1";
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), spec.into());
        let parser = create_parser(&handler, spec);
        let pair = LolaParser::parse(Rule::OutputStream, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_output(pair);
        assert_eq!(format!("{}", ast), spec)
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
        let spec = "trigger in ≠ out \"some message\"";
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), spec.into());
        let parser = create_parser(&handler, spec);
        let pair = LolaParser::parse(Rule::Trigger, spec)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.parse_trigger(pair);
        assert_eq!(format!("{}", ast), "trigger in ≠ out \"some message\"")
    }

    #[test]
    fn parse_expression() {
        let content = "in + 1";
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), content.into());
        let parser = create_parser(&handler, content);
        let expr = LolaParser::parse(Rule::Expr, content)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.build_expression_ast(expr.into_inner());
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_expression_precedence() {
        let content = "(a ∨ b ∧ c)";
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), content.into());
        let parser = create_parser(&handler, content);
        let expr = LolaParser::parse(Rule::Expr, content)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.build_expression_ast(expr.into_inner());
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_missing_closing_parenthesis() {
        let content = "(a ∨ b ∧ c";
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), content.into());
        let parser = create_parser(&handler, content);
        let expr = LolaParser::parse(Rule::Expr, content)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let ast = parser.build_expression_ast(expr.into_inner());
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn build_simple_ast() {
        let spec = "input in: Int\noutput out: Int := in\ntrigger in ≠ out\n";
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
        let spec = "output s: Bool := (true ∨ true)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_optional_type() {
        let spec = "output s: Bool? := (false ∨ true)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_default() {
        let spec = "output s: Int := s.offset(by: -1).defaults(to: (3 * 4))\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_hold() {
        let spec = "output s: Int := s.offset(by: -1).hold().defaults(to: 3 * 4)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ternary_expression() {
        let spec = "input in: Int\noutput s: Int := if in = 3 then 4 else in + 2\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_function_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := nroot(1, sin(1, in))\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger() {
        let spec = "input in: Int\ntrigger in > 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger_extend() {
        let spec = "trigger @1Hz in > 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger_message() {
        let spec = "trigger in > 5 \"test trigger\"\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
        assert_eq!(ast.trigger[0].message, Some("test trigger".to_string()));
    }

    #[test]
    fn build_trigger_message_with_info_streams() {
        let spec = "trigger in > 5 \"test trigger\" (i, o, x)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
        assert_eq!(ast.trigger[0].info_streams.len(), 3);
        assert_eq!(ast.trigger[0].info_streams[0].name, "i".to_string());
        assert_eq!(ast.trigger[0].info_streams[1].name, "o".to_string());
        assert_eq!(ast.trigger[0].info_streams[2].name, "x".to_string());
    }

    #[test]
    fn build_trigger_message_with_info_streams_faulty() {
        let spec = "trigger in > 5 (i, o, x)\n";
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), spec.into());
        let parser = create_parser(&handler, spec);
        assert!(matches!(parser.parse_spec(), Err(_)));
    }

    #[test]
    fn build_complex_expression() {
        let spec =
            "output s: Double := if !((s.offset(by: -1).defaults(to: (3 * 4)) + -4) = 12) ∨ true = false then 2.0 else 4.1\n";
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
        let spec = "output s (a: B, c: D): E := 3\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_termination_spec() {
        let spec = "output s (a: Int): Int close s > 10 := 3\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_tuple_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := (1, in.0).1\n";
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
        let spec = "import math\ninput a: Int32\ninput b: Int32\noutput maxres: Int32 := max<Int32>(a, b)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call() {
        let spec = "output count := count.offset(-1).default(0) + 1\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call_with_param() {
        let spec = "output count := count.offset<Int8>(-1).default(0) + 1\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_realtime_offset() {
        let spec = "output a := b.offset(by: -1s)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_future_offset() {
        let spec = "output a := b.offset(by: 1)\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_function_argument_name() {
        let spec = "output a := b.hold().defaults(to: 0)\n";
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
    fn regression71() {
        let spec = "output outputstream := 42 output c := outputstream";
        let ast = parse(spec);
        cmp_ast_spec(&ast, "output outputstream := 42\noutput c := outputstream\n");
    }

    #[test]
    fn parse_bitwise() {
        let spec = "output x := 1 ^ 0 & 23123 | 111\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn spawn_no_target() {
        let spec = "output x spawn if true := 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_template_spec() {
        let spec = "output x (y: Int8) @ 1Hz spawn with (42) if true filter y = 1337 close false := 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn spawn_no_target_no_condition() {
        let spec = "output x spawn := 5\n";
        let handler = Handler::new(PathBuf::from("parser/test".to_string()), spec.into());
        let parser = create_parser(&handler, spec);
        parser.parse_spec().unwrap_or_else(|e| panic!("{}", e));
        assert_eq!(handler.emitted_errors(), 1);
    }

    #[test]
    fn spawn_with_pacing() {
        let spec = "output x spawn @3Hz with (x) if true := 5\n";
        let ast = parse(spec);
        cmp_ast_spec(&ast, spec);
    }
}
