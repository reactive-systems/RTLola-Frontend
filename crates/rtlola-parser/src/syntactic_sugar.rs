use std::collections::HashSet;
use std::hash::Hash;
use std::rc::Rc;

// List for all syntactic sugar transformer
mod aggregation_method;
mod delta;
mod implication;
mod last;
mod mirror;
mod offset_or;
use aggregation_method::AggrMethodToWindow;
use delta::Delta;
use last::Last;
use mirror::Mirror as SynSugMirror;

use self::implication::Implication;
use self::offset_or::OffsetOr;
use crate::ast::{
    CloseSpec, EvalSpec, Expression, ExpressionKind, Input, Mirror as AstMirror, NodeId, Output, RtLolaAst, SpawnSpec,
};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum ChangeInstruction {
    #[allow(dead_code)] // currently unused
    ReplaceExpr(NodeId, Expression),
    RemoveStream(NodeId),
    AddOutput(Box<Output>),
}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum LocalChangeInstruction {
    ReplaceExpr(Expression),
}

/// The ChangeSet collects change instructions for syntactic sugar removal.
///
/// Desugarization could replace the current expression or add global changes, like outsource to new streams.
/// Adding change sets collects all global changes and allows at most one local change to be present.
#[derive(Debug, Clone)]
struct ChangeSet {
    _local_applied_flag: bool,
    local_instructions: Option<LocalChangeInstruction>,
    global_instructions: HashSet<ChangeInstruction>,
}

/// Syntactic Sugar has to implement this trait and override methods if needed.
///
/// The transformer gets every single expression passed once, in an bottom up order. Afterwards every top level stream object is passed once.
/// The [ChangeSet] can hold a single local change, a change that immediately has to be applied to the expression passed in [SynSugar::desugarize_expr],
/// and an arbitrary number of changes that will be applied after an iteration, which is identified by the [NodeId] of the object it wants to replace.
#[allow(unused_variables)] // allow unused arguments in the default implementation without changing the name
trait SynSugar {
    /// Desugars a single expression.  Provided [RtLolaAst] and [Expression] are for reference, not modification.
    ///
    /// # Requirements
    /// * Ids may NEVER be re-used.  Always generate new ones with ast.next_id() or increase the prime-counter of an existing [NodeId].
    /// * When creating new nodes with a span, do not re-use the span of the old node.  Instead, create an indirect span refering to the old one.
    fn desugarize_expr<'a>(&self, exp: &'a Expression, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }

    /// Desugars a single output stream.  Provided [RtLolaAst] and [Output] are for reference, not modification.
    ///
    /// # Requirements
    /// * Ids may NEVER be re-used.  Always generate new ones with ast.next_id() or increase the prime-counter of an existing [NodeId].
    /// * When creating new nodes with a span, do not re-use the span of the old node.  Instead, create an indirect span refering to the old one.
    fn desugarize_stream_out<'a>(&self, stream: &'a Output, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }
    /// Desugars a single input stream.  Provided [RtLolaAst] and [Input] are for reference, not modification.
    ///
    /// # Requirements
    /// * Ids may NEVER be re-used.  Always generate new ones with ast.next_id() or increase the prime-counter of an existing [NodeId].
    /// * When creating new nodes with a span, do not re-use the span of the old node.  Instead, create an indirect span refering to the old one.
    fn desugarize_stream_in<'a>(&self, stream: &'a Input, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }
    /// Desugars a single mirror stream.  Provided [RtLolaAst] and [AstMirror] are for reference, not modification.
    ///
    /// # Requirements
    /// * Ids may NEVER be re-used.  Always generate new ones with ast.next_id() or increase the prime-counter of an existing [NodeId].
    /// * When creating new nodes with a span, do not re-use the span of the old node.  Instead, create an indirect span refering to the old one.
    fn desugarize_stream_mirror<'a>(&self, stream: &'a AstMirror, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }
}

/// The container for all syntactic sugar structs.
///
/// Contains the logic on how and when to remove syntactic sugar changes.
/// A transformer just needs to implement the SynSugar trait and be registered in the internal vector, constructed in [Desugarizer::all].
#[allow(missing_debug_implementations)] // Contains no relevant fields, only logic. Trait object vectors are hard to print.
pub struct Desugarizer {
    sugar_transformers: Vec<Box<dyn SynSugar>>,
}

impl Desugarizer {
    /// Constructs a Desugarizer with all registered SynSugar transformations.
    ///
    /// All transformations registered in the internal vector will be applied on the ast.
    /// New structs have to be added in this function.
    pub fn all() -> Self {
        let all_transformers: Vec<Box<dyn SynSugar>> = vec![
            Box::new(Implication {}),
            Box::new(AggrMethodToWindow {}),
            Box::new(Last {}),
            Box::new(SynSugMirror {}),
            Box::new(Delta {}),
            Box::new(OffsetOr {}),
        ];
        Self {
            sugar_transformers: all_transformers,
        }
    }

    /// Before calling this, all others Rcs on Ast nodes MUST be dropped.
    ///
    /// # panics
    /// If an Rc has a strong count greater than one.
    pub fn remove_syn_sugar(&self, mut ast: RtLolaAst) -> RtLolaAst {
        while {
            // magic rust do-while loop
            let (new_ast, change_flag) = self.desugarize_fix_point(ast);
            ast = new_ast;
            change_flag
        } {} // do not remove! magic rust do-while loop
        ast
    }

    fn desugarize_fix_point(&self, mut ast: RtLolaAst) -> (RtLolaAst, bool) {
        let mut change_flag = false;
        for current_sugar in self.sugar_transformers.iter() {
            let mut change_set = ChangeSet::empty();

            for mirror in ast.mirrors.iter() {
                change_set += self.desugarize_mirror(mirror, &ast, current_sugar);
            }

            for ix in 0..ast.outputs.len() {
                let out = &ast.outputs[ix];
                let out_clone: Output = Output::clone(out);
                let Output { spawn, eval, close, .. } = out_clone;
                let new_spawn_spec = if let Some(spawn_spec) = spawn {
                    let SpawnSpec {
                        expression,
                        condition,
                        annotated_pacing,
                        id,
                        span,
                    } = spawn_spec;
                    let expression = expression.map(|expr| {
                        let (new_expr, spawn_cs) = Self::desugarize_expression(expr, &ast, current_sugar);
                        change_set += spawn_cs;
                        new_expr
                    });
                    let condition = condition.map(|expr| {
                        let (new_expr, spawn_cond_cs) = Self::desugarize_expression(expr, &ast, current_sugar);
                        change_set += spawn_cond_cs;
                        new_expr
                    });
                    Some(SpawnSpec {
                        expression,
                        condition,
                        annotated_pacing,
                        id,
                        span,
                    })
                } else {
                    None
                };

                let transformed_eval = eval
                    .into_iter()
                    .flat_map(|eval_spec| {
                        let EvalSpec {
                            annotated_pacing,
                            condition,
                            eval_expression,
                            id,
                            span,
                        } = eval_spec;
                        let new_eval = eval_expression.map(|e| {
                            let (res, eval_cs) = Self::desugarize_expression(e, &ast, current_sugar);
                            change_set += eval_cs;
                            res
                        });
                        let new_condition = condition.map(|e| {
                            let (res, cond_cs) = Self::desugarize_expression(e, &ast, current_sugar);
                            change_set += cond_cs;
                            res
                        });
                        Some(EvalSpec {
                            annotated_pacing,
                            condition: new_condition,
                            eval_expression: new_eval,
                            id,
                            span,
                        })
                    })
                    .collect();

                let new_close_spec = if let Some(close_spec) = close {
                    let CloseSpec {
                        condition,
                        id,
                        span,
                        annotated_pacing,
                    } = close_spec;
                    let (new_condition, close_cs) = Self::desugarize_expression(condition, &ast, current_sugar);
                    change_set += close_cs;
                    Some(CloseSpec {
                        condition: new_condition,
                        id,
                        span,
                        annotated_pacing,
                    })
                } else {
                    None
                };

                let new_out = Output {
                    spawn: new_spawn_spec,
                    eval: transformed_eval,
                    close: new_close_spec,
                    ..out_clone
                };
                ast.outputs[ix] = Rc::new(new_out);
            }
            for input in ast.inputs.iter() {
                change_set += self.desugarize_input(input, &ast, current_sugar);
            }

            for output in ast.outputs.iter() {
                change_set += self.desugarize_output(output, &ast, current_sugar);
            }

            change_flag |= change_set._local_applied_flag || !change_set.global_instructions.is_empty();
            ast = self.apply_global_changes(change_set, ast);
        }
        (ast, change_flag)
    }

    fn apply_global_changes(&self, c_s: ChangeSet, mut ast: RtLolaAst) -> RtLolaAst {
        c_s.global_iter().for_each(|ci| {
            match ci {
                ChangeInstruction::AddOutput(o) => {
                    ast.outputs.push(Rc::new(*o));
                },
                ChangeInstruction::RemoveStream(id) => {
                    if let Some(idx) = ast.outputs.iter().position(|o| o.id == id) {
                        assert_eq!(Rc::strong_count(&ast.outputs[idx]), 1);
                        ast.outputs.remove(idx);
                    } else if let Some(idx) = ast.inputs.iter().position(|o| o.id == id) {
                        assert_eq!(Rc::strong_count(&ast.inputs[idx]), 1);
                        ast.inputs.remove(idx);
                    } else if let Some(idx) = ast.mirrors.iter().position(|o| o.id == id) {
                        assert_eq!(Rc::strong_count(&ast.mirrors[idx]), 1);
                        ast.mirrors.remove(idx);
                    } else {
                        debug_assert!(false, "id in changeset does not belong to any stream");
                    }
                },
                ChangeInstruction::ReplaceExpr(_, expr) => {
                    for ix in 0..ast.outputs.len() {
                        let out = &ast.outputs[ix];
                        let out_clone: Output = Output::clone(out);
                        let Output { spawn, eval, close, .. } = out_clone;
                        let new_spawn_spec = if let Some(spawn_spec) = spawn {
                            let SpawnSpec {
                                expression,
                                condition,
                                annotated_pacing,
                                id,
                                span,
                            } = spawn_spec;
                            let expression = expression
                                .map(|tar_expression| Self::apply_expr_global_change(id, &expr, &tar_expression));
                            let condition = condition
                                .map(|cond_expression| Self::apply_expr_global_change(id, &expr, &cond_expression));
                            Some(SpawnSpec {
                                expression,
                                condition,
                                annotated_pacing,
                                id,
                                span,
                            })
                        } else {
                            None
                        };

                        let new_eval_spec = eval
                            .into_iter()
                            .flat_map(|eval_spec| {
                                let EvalSpec {
                                    eval_expression,
                                    condition,
                                    annotated_pacing,
                                    id,
                                    span,
                                } = eval_spec;
                                let eval_expression =
                                    eval_expression.map(|e| Self::apply_expr_global_change(id, &expr, &e));
                                let condition = condition.map(|e| Self::apply_expr_global_change(id, &expr, &e));
                                Some(EvalSpec {
                                    annotated_pacing,
                                    condition,
                                    eval_expression,
                                    id,
                                    span,
                                })
                            })
                            .collect();

                        let new_close_spec = if let Some(close_spec) = close {
                            let CloseSpec {
                                condition,
                                id,
                                span,
                                annotated_pacing,
                            } = close_spec;
                            let new_condition = Self::apply_expr_global_change(id, &expr, &condition);
                            Some(CloseSpec {
                                condition: new_condition,
                                id,
                                span,
                                annotated_pacing,
                            })
                        } else {
                            None
                        };

                        let new_out = Output {
                            spawn: new_spawn_spec,
                            eval: new_eval_spec,
                            close: new_close_spec,
                            ..out_clone
                        };
                        ast.outputs[ix] = Rc::new(new_out);
                    }
                },
            }
        });

        ast
    }

    fn apply_expr_global_change(target_id: NodeId, new_expr: &Expression, ast_expr: &Expression) -> Expression {
        if ast_expr.id == target_id {
            return new_expr.clone();
        }
        let span = ast_expr.span;

        use ExpressionKind::*;
        match &ast_expr.kind {
            Lit(_) | Ident(_) | MissingExpression => ast_expr.clone(),
            Unary(op, inner) => {
                Expression {
                    kind: Unary(
                        *op,
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, inner)),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Field(inner, ident) => {
                Expression {
                    kind: Field(
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, inner)),
                        ident.clone(),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            StreamAccess(inner, acc_kind) => {
                Expression {
                    kind: StreamAccess(
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, inner)),
                        *acc_kind,
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Offset(inner, offset) => {
                Expression {
                    kind: Offset(
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, inner)),
                        *offset,
                    ),
                    span,
                    ..*ast_expr
                }
            },
            ParenthesizedExpression(lp, inner, rp) => {
                Expression {
                    kind: ParenthesizedExpression(
                        lp.clone(),
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, inner)),
                        rp.clone(),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Binary(bin_op, left, right) => {
                Expression {
                    kind: Binary(
                        *bin_op,
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, left)),
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, right)),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Default(left, right) => {
                Expression {
                    kind: Default(
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, left)),
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, right)),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            DiscreteWindowAggregation {
                expr: left,
                duration: right,
                wait,
                aggregation,
                ..
            } => {
                Expression {
                    kind: DiscreteWindowAggregation {
                        expr: Box::new(Self::apply_expr_global_change(target_id, new_expr, left)),
                        duration: Box::new(Self::apply_expr_global_change(target_id, new_expr, right)),
                        wait: *wait,
                        aggregation: *aggregation,
                    },
                    span,
                    ..*ast_expr
                }
            },
            SlidingWindowAggregation {
                expr: left,
                duration: right,
                wait,
                aggregation,
            } => {
                Expression {
                    kind: SlidingWindowAggregation {
                        expr: Box::new(Self::apply_expr_global_change(target_id, new_expr, left)),
                        duration: Box::new(Self::apply_expr_global_change(target_id, new_expr, right)),
                        wait: *wait,
                        aggregation: *aggregation,
                    },
                    span,
                    ..*ast_expr
                }
            },
            InstanceAggregation {
                expr,
                selection,
                aggregation,
            } => {
                Expression {
                    kind: InstanceAggregation {
                        expr: Box::new(Self::apply_expr_global_change(target_id, new_expr, expr)),
                        selection: *selection,
                        aggregation: *aggregation,
                    },
                    span,
                    ..*ast_expr
                }
            },
            Ite(condition, normal, alternative) => {
                Expression {
                    kind: Ite(
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, condition)),
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, normal)),
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, alternative)),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Tuple(entries) => {
                Expression {
                    kind: Tuple(
                        entries
                            .iter()
                            .map(|t_expr| Self::apply_expr_global_change(target_id, new_expr, t_expr))
                            .collect(),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Function(name, types, entries) => {
                Expression {
                    kind: Function(
                        name.clone(),
                        types.clone(),
                        entries
                            .iter()
                            .map(|t_expr| Self::apply_expr_global_change(target_id, new_expr, t_expr))
                            .collect(),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Method(base, name, types, arguments) => {
                Expression {
                    kind: Method(
                        Box::new(Self::apply_expr_global_change(target_id, new_expr, base)),
                        name.clone(),
                        types.clone(),
                        arguments
                            .iter()
                            .map(|t_expr| Self::apply_expr_global_change(target_id, new_expr, t_expr))
                            .collect(),
                    ),
                    span,
                    ..*ast_expr
                }
            },
        }
    }

    /// Climbs along the expression and applies desugarizers and their local changes, while collecting global changes.
    ///
    /// LocalChangeInstructions are directly replied and never returned.
    #[allow(clippy::borrowed_box)]
    fn desugarize_expression(
        ast_expr: Expression,
        ast: &RtLolaAst,
        current_sugar: &Box<dyn SynSugar>,
    ) -> (Expression, ChangeSet) {
        let mut return_cs = ChangeSet::empty();
        use ExpressionKind::*;
        let Expression { kind, id, span } = ast_expr;
        let new_expr = match kind {
            Lit(_) | Ident(_) | MissingExpression => Expression { kind, id, span },
            Unary(op, inner) => {
                let (inner, cs) = Self::desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: Unary(op, Box::new(inner)),
                    span,
                    id,
                }
            },
            Field(inner, ident) => {
                let (inner, cs) = Self::desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: Field(Box::new(inner), ident),
                    span,
                    id,
                }
            },
            StreamAccess(inner, acc_kind) => {
                let (inner, cs) = Self::desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: StreamAccess(Box::new(inner), acc_kind),
                    span,
                    id,
                }
            },
            Offset(inner, offset) => {
                let (inner, cs) = Self::desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: Offset(Box::new(inner), offset),
                    span,
                    id,
                }
            },
            ParenthesizedExpression(lp, inner, rp) => {
                let (inner, cs) = Self::desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: ParenthesizedExpression(lp, Box::new(inner), rp),
                    span,
                    id,
                }
            },
            Binary(bin_op, left, right) => {
                let (left, lcs) = Self::desugarize_expression(*left, ast, current_sugar);
                return_cs += lcs;
                let (right, rcs) = Self::desugarize_expression(*right, ast, current_sugar);
                return_cs += rcs;
                Expression {
                    kind: Binary(bin_op, Box::new(left), Box::new(right)),
                    span,
                    id,
                }
            },
            Default(left, right) => {
                let (left, lcs) = Self::desugarize_expression(*left, ast, current_sugar);
                return_cs += lcs;
                let (right, rcs) = Self::desugarize_expression(*right, ast, current_sugar);
                return_cs += rcs;
                Expression {
                    kind: Default(Box::new(left), Box::new(right)),
                    span,
                    id,
                }
            },
            DiscreteWindowAggregation {
                expr: left,
                duration: right,
                wait,
                aggregation,
                ..
            } => {
                let (expr, ecs) = Self::desugarize_expression(*left, ast, current_sugar);
                return_cs += ecs;
                let (dur, dcs) = Self::desugarize_expression(*right, ast, current_sugar);
                return_cs += dcs;
                Expression {
                    kind: DiscreteWindowAggregation {
                        expr: Box::new(expr),
                        duration: Box::new(dur),
                        wait,
                        aggregation,
                    },
                    span,
                    id,
                }
            },
            SlidingWindowAggregation {
                expr: left,
                duration: right,
                wait,
                aggregation,
            } => {
                let (expr, ecs) = Self::desugarize_expression(*left, ast, current_sugar);
                return_cs += ecs;
                let (dur, dcs) = Self::desugarize_expression(*right, ast, current_sugar);
                return_cs += dcs;
                Expression {
                    kind: SlidingWindowAggregation {
                        expr: Box::new(expr),
                        duration: Box::new(dur),
                        wait,
                        aggregation,
                    },
                    span,
                    id,
                }
            },
            InstanceAggregation {
                expr,
                selection,
                aggregation,
            } => {
                let (expr, ecs) = Self::desugarize_expression(*expr, ast, current_sugar);
                return_cs += ecs;
                Expression {
                    kind: InstanceAggregation {
                        expr: Box::new(expr),
                        selection,
                        aggregation,
                    },
                    span,
                    id,
                }
            },
            Ite(condition, normal, alternative) => {
                let (condition, ccs) = Self::desugarize_expression(*condition, ast, current_sugar);
                return_cs += ccs;
                let (normal, ncs) = Self::desugarize_expression(*normal, ast, current_sugar);
                return_cs += ncs;
                let (alternative, acs) = Self::desugarize_expression(*alternative, ast, current_sugar);
                return_cs += acs;
                Expression {
                    kind: Ite(Box::new(condition), Box::new(normal), Box::new(alternative)),
                    span,
                    id,
                }
            },
            Tuple(entries) => {
                let (v_expr, v_cs): (Vec<Expression>, Vec<ChangeSet>) = entries
                    .into_iter()
                    .map(|t_expr| Self::desugarize_expression(t_expr, ast, current_sugar))
                    .unzip();
                return_cs += v_cs.into_iter().fold(ChangeSet::empty(), |acc, x| acc + x);
                Expression {
                    kind: Tuple(v_expr),
                    span,
                    id,
                }
            },
            Function(name, types, entries) => {
                let (v_expr, v_cs): (Vec<Expression>, Vec<ChangeSet>) = entries
                    .into_iter()
                    .map(|t_expr| Self::desugarize_expression(t_expr, ast, current_sugar))
                    .unzip();
                return_cs += v_cs.into_iter().fold(ChangeSet::empty(), |acc, x| acc + x);
                Expression {
                    kind: Function(name, types, v_expr),
                    span,
                    id,
                }
            },
            Method(base, name, types, arguments) => {
                let (base_expr, ecs) = Self::desugarize_expression(*base, ast, current_sugar);
                return_cs += ecs;
                let (v_expr, v_cs): (Vec<Expression>, Vec<ChangeSet>) = arguments
                    .into_iter()
                    .map(|t_expr| Self::desugarize_expression(t_expr, ast, current_sugar))
                    .unzip();
                return_cs += v_cs.into_iter().fold(ChangeSet::empty(), |acc, x| acc + x);
                Expression {
                    kind: Method(Box::new(base_expr), name, types, v_expr),
                    span,
                    id,
                }
            },
        };

        // apply transformation on current expression and replace if local change needed
        let mut current_level_cs = current_sugar.desugarize_expr(&new_expr, ast);
        let return_expr =
            if let Some(LocalChangeInstruction::ReplaceExpr(replace_expr)) = current_level_cs.extract_local_change() {
                replace_expr
            } else {
                new_expr
            };
        let final_cs = current_level_cs + return_cs;

        (return_expr, final_cs)
    }

    #[allow(clippy::borrowed_box)]
    fn desugarize_input(&self, input: &Input, ast: &RtLolaAst, current_sugar: &Box<dyn SynSugar>) -> ChangeSet {
        current_sugar.desugarize_stream_in(input, ast)
    }

    #[allow(clippy::borrowed_box)]
    fn desugarize_mirror(&self, mirror: &AstMirror, ast: &RtLolaAst, current_sugar: &Box<dyn SynSugar>) -> ChangeSet {
        current_sugar.desugarize_stream_mirror(mirror, ast)
    }

    #[allow(clippy::borrowed_box)]
    fn desugarize_output(&self, output: &Output, ast: &RtLolaAst, current_sugar: &Box<dyn SynSugar>) -> ChangeSet {
        current_sugar.desugarize_stream_out(output, ast)
    }
}

impl Default for Desugarizer {
    fn default() -> Self {
        Self::all()
    }
}

impl ChangeSet {
    /// Construct a new empty ChangeSet.
    fn empty() -> Self {
        Self {
            _local_applied_flag: false,
            local_instructions: None,
            global_instructions: HashSet::new(),
        }
    }

    #[allow(dead_code)] // currently unused
    /// Adds a stream to the Ast.
    pub(crate) fn add_output(stream: Output) -> Self {
        let mut cs = ChangeSet::empty();
        cs.global_instructions
            .insert(ChangeInstruction::AddOutput(Box::new(stream)));
        cs
    }

    #[allow(dead_code)] // currently unused
    /// Replaces the expression with id 'target_id' with the given expression. Performs global ast search.
    pub(crate) fn replace_expression(target_id: NodeId, expr: Expression) -> Self {
        let mut cs = Self::empty();
        cs.global_instructions
            .insert(ChangeInstruction::ReplaceExpr(target_id, expr));
        cs
    }

    #[allow(dead_code)] // currently unused
    /// Removes the stream with id 'target_id'. Performs global ast search.
    pub(crate) fn remove_stream(target_id: NodeId) -> Self {
        let mut cs = Self::empty();
        cs.global_instructions
            .insert(ChangeInstruction::RemoveStream(target_id));
        cs
    }

    /// Replaces the current expression.
    /// Should only be called in <T: SynSugar> T.desugarize_expr(...)
    /// Wanted local changes will not be applied if used in other desugarize functions!
    pub(crate) fn replace_current_expression(expr: Expression) -> Self {
        let mut cs = Self::empty();
        cs.local_instructions = Some(LocalChangeInstruction::ReplaceExpr(expr));
        cs
    }

    /// Replaces the stream with the given NodeId with the new stream. Performs global ast search.
    pub(crate) fn replace_stream(target_stream: NodeId, new_stream: Output) -> Self {
        let mut cs = ChangeSet::empty();
        cs.global_instructions
            .insert(ChangeInstruction::RemoveStream(target_stream));
        cs.global_instructions
            .insert(ChangeInstruction::AddOutput(Box::new(new_stream)));
        cs
    }

    /// Provides an [Iterator] over all global changes.
    fn global_iter<'a>(self) -> Box<dyn Iterator<Item = ChangeInstruction> + 'a> {
        Box::new(self.global_instructions.into_iter())
    }

    /// Internal use: extracts the wanted local change and removes it from the struct.
    fn extract_local_change(&mut self) -> Option<LocalChangeInstruction> {
        let mut ret = None;
        std::mem::swap(&mut self.local_instructions, &mut ret);
        ret
    }
}

impl std::ops::Add<ChangeSet> for ChangeSet {
    type Output = ChangeSet;

    fn add(self, rhs: Self) -> Self {
        let set = self
            .global_instructions
            .union(&rhs.global_instructions)
            .cloned()
            .collect();
        assert!(self.local_instructions.is_none() || rhs.local_instructions.is_none());
        let local = self.local_instructions.or(rhs.local_instructions);
        Self {
            _local_applied_flag: self._local_applied_flag || rhs._local_applied_flag,
            local_instructions: local,
            global_instructions: set,
        }
    }
}

impl std::ops::AddAssign<ChangeSet> for ChangeSet {
    fn add_assign(&mut self, rhs: Self) {
        let set = self
            .global_instructions
            .union(&rhs.global_instructions)
            .cloned()
            .collect();
        assert!(self.local_instructions.is_none() || rhs.local_instructions.is_none());
        let local = self.local_instructions.clone().or(rhs.local_instructions);
        *self = Self {
            _local_applied_flag: self._local_applied_flag || rhs._local_applied_flag,
            local_instructions: local,
            global_instructions: set,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Ident, LitKind, Literal, Offset, UnOp, WindowOperation};

    #[test]
    fn test_impl_simpl_replace() {
        let spec = "input a:Bool\ninput b:Bool\noutput c eval with a -> b".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind.clone();
        let inner_kind = if let ExpressionKind::Binary(op, lhs, _rhs) = out_kind {
            assert!(matches!(op, BinOp::Or));
            lhs.kind
        } else {
            unreachable!()
        };
        assert!(matches!(inner_kind, ExpressionKind::Unary(UnOp::Not, _)));
    }

    #[test]
    fn test_impl_nested_replace() {
        let spec = "input a:Bool\ninput b:Bool\ninput c:Bool\noutput d eval with a -> b -> c".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind.clone();
        let inner_kind = if let ExpressionKind::Binary(op, lhs, rhs) = out_kind {
            assert!(matches!(op, BinOp::Or));
            let inner = if let ExpressionKind::Unary(op, inner) = lhs.kind {
                assert!(matches!(op, UnOp::Not));
                inner
            } else {
                unreachable!()
            };
            assert_eq!(inner.to_string(), "a");
            rhs.kind
        } else {
            unreachable!();
        };
        let inner = if let ExpressionKind::Binary(op, lhs, rhs) = inner_kind {
            assert!(matches!(op, BinOp::Or));
            let inner = if let ExpressionKind::Unary(op, inner) = lhs.kind {
                assert!(matches!(op, UnOp::Not));
                inner
            } else {
                unreachable!()
            };
            assert_eq!(inner.to_string(), "b");
            rhs
        } else {
            unreachable!()
        };
        assert_eq!(inner.to_string(), "c");
    }

    #[test]
    fn test_offsetor_replace() {
        let spec = "output x eval @5Hz with x.offset(by: -4, or: 5.0)".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind.clone();
        let inner_kind = if let ExpressionKind::Default(inner, default) = out_kind {
            assert!(matches!(default.kind, ExpressionKind::Lit(_)));
            inner.kind
        } else {
            unreachable!()
        };
        assert!(matches!(inner_kind, ExpressionKind::Offset(_, Offset::Discrete(-4))));
    }

    #[test]
    fn test_offsetor_replace_nested() {
        let spec = "output x eval @5Hz with -x.offset(by: -4, or: x.offset(by: -1, or: 5))".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind.clone();
        let inner_kind = if let ExpressionKind::Unary(UnOp::Neg, inner) = out_kind {
            inner.kind
        } else {
            unreachable!()
        };
        let (inner_kind, default_kind) = if let ExpressionKind::Default(inner, default) = inner_kind {
            (inner.kind, default.kind)
        } else {
            unreachable!()
        };
        assert!(matches!(inner_kind, ExpressionKind::Offset(_, Offset::Discrete(-4))));
        let inner_kind = if let ExpressionKind::Default(inner, default) = default_kind {
            assert!(matches!(default.kind, ExpressionKind::Lit(_)));
            inner.kind
        } else {
            unreachable!()
        };
        assert!(matches!(inner_kind, ExpressionKind::Offset(_, Offset::Discrete(-1))));
    }

    #[test]
    fn test_aggr_replace() {
        let spec = "output x eval @5Hz with x.count(6s)".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        assert!(matches!(
            ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind,
            ExpressionKind::SlidingWindowAggregation {
                aggregation: WindowOperation::Count,
                ..
            }
        ));
    }

    #[test]
    fn test_aggr_replace_nested() {
        let spec = "output x eval @ 5hz with -x.sum(6s)".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind.clone();
        assert!(matches!(out_kind, ExpressionKind::Unary(UnOp::Neg, _)));
        let inner_kind = if let ExpressionKind::Unary(UnOp::Neg, inner) = out_kind {
            inner.kind
        } else {
            unreachable!()
        };
        assert!(matches!(
            inner_kind,
            ExpressionKind::SlidingWindowAggregation {
                aggregation: WindowOperation::Sum,
                ..
            }
        ));
    }

    #[test]
    fn test_aggr_replace_multiple() {
        let spec = "output x eval @5hz with x.avg(5s) - x.integral(2.5s)".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind.clone();
        assert!(matches!(out_kind, ExpressionKind::Binary(BinOp::Sub, _, _)));
        let (left, right) = if let ExpressionKind::Binary(BinOp::Sub, left, right) = out_kind {
            (left.kind, right.kind)
        } else {
            unreachable!()
        };
        assert!(matches!(
            left,
            ExpressionKind::SlidingWindowAggregation {
                aggregation: WindowOperation::Average,
                ..
            }
        ));
        assert!(matches!(
            right,
            ExpressionKind::SlidingWindowAggregation {
                aggregation: WindowOperation::Integral,
                ..
            }
        ));
    }

    #[test]
    fn test_last_replace() {
        let spec = "output x eval @5hz with x.last(or: 3)".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].eval[0].clone().eval_expression.unwrap().kind.clone();
        let (access, dft) = if let ExpressionKind::Default(access, dft) = out_kind {
            (access.kind, dft.kind)
        } else {
            panic!("Last should result in a top-level default access with its argument as default value")
        };
        assert!(
            matches!(
                &dft,
                &ExpressionKind::Lit(Literal {
                    kind: LitKind::Numeric(ref x, None),
                    ..
                }) if x == &String::from("3")
            ),
            "The argument of last should be the default expression."
        );
        let stream = if let ExpressionKind::Offset(stream, Offset::Discrete(-1)) = access {
            stream
        } else {
            panic!("expected an offset expression, but found {:?}", access);
        };

        assert!(
            matches!(*stream, Expression { kind: ExpressionKind::Ident(Ident { name, .. }), ..} if name == String::from("x") )
        );
    }

    #[test]
    fn test_delta_replace() {
        let spec = "output y eval with delta(x,dft:0)".to_string();
        let expected = "output y eval with x - x.offset(by: -1).defaults(to: 0)";
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        assert_eq!(expected, format!("{}", ast).trim());
    }

    #[test]
    fn test_delta_replace_float() {
        let spec = "output y eval with delta(x, or: 0.0)".to_string();
        let expected = "output y eval with x - x.offset(by: -1).defaults(to: 0.0)";
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        assert_eq!(expected, format!("{}", ast).trim());
    }

    #[test]
    fn test_mirror_replace() {
        let spec = "output x eval with 3 \noutput y mirrors x when x > 5".to_string();
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        assert_eq!(ast.outputs.len(), 2);
        assert!(ast.mirrors.is_empty());
        let new = &ast.outputs[1];
        let target = &ast.outputs[0];
        assert_eq!(target.name().unwrap().name, "x");
        assert_eq!(new.name().unwrap().name, "y");
        assert_eq!(new.annotated_type, target.annotated_type);
        assert_eq!(
            new.eval[0].clone().annotated_pacing,
            target.eval[0].clone().annotated_pacing
        );
        assert_eq!(new.close, target.close);
        assert_eq!(
            new.eval[0].eval_expression.clone(),
            target.eval[0].eval_expression.clone()
        );
        assert!(new.eval[0].clone().condition.is_some());
        assert!(matches!(
            new.eval[0].clone().condition.as_ref().unwrap(),
            Expression {
                kind: ExpressionKind::Binary(..),
                ..
            }
        ));
        assert_eq!(new.params, target.params);
        assert_eq!(new.spawn, target.spawn);
    }

    #[test]
    fn test_mirror_replace_str_cmp() {
        let spec = "output x eval with 3 \noutput y mirrors x when x > 5".to_string();
        let expected = "output x eval with 3\noutput y eval when x > 5 with 3";
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        assert_eq!(expected, format!("{}", ast).trim());
    }

    #[test]
    fn test_mirror_replace_multiple_eval() {
        let spec = "output x eval when a > 0 with 3 eval when a < 0 with -3\noutput y mirrors x when x > 5".to_string();
        let expected = "output x eval when a > 0 with 3 eval when a < 0 with -3\noutput y eval when a > 0 ∧ x > 5 with 3 eval when a < 0 ∧ x > 5 with -3";
        let ast = crate::parse(&crate::ParserConfig::for_string(spec)).unwrap();
        assert_eq!(expected, format!("{}", ast).trim());
    }
}
