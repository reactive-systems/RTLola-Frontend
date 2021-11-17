use std::collections::HashSet;
use std::hash::Hash;
use std::rc::Rc;

// List for all syntactic sugar transformer
mod aggregation_method;
mod delta;
mod last;
mod mirror;
use aggregation_method::AggrMethodToWindow;
use delta::Delta;
use last::Last;
use mirror::Mirror as SynSugMirror;

use crate::ast::{
    CloseSpec, Expression, ExpressionKind, FilterSpec, Input, Mirror as AstMirror, NodeId, Output, RtLolaAst,
    SpawnSpec, Trigger,
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
    /// Desugars a single trigger.  Provided [RtLolaAst] and [Trigger] are for reference, not modification.
    ///
    /// # Requirements
    /// * Ids may NEVER be re-used.  Always generate new ones with ast.next_id() or increase the prime-counter of an existing [NodeId].
    /// * When creating new nodes with a span, do not re-use the span of the old node.  Instead, create an indirect span refering to the old one.
    fn desugarize_stream_trigger<'a>(&self, stream: &'a Trigger, ast: &'a RtLolaAst) -> ChangeSet {
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
            Box::new(AggrMethodToWindow {}),
            Box::new(Last {}),
            Box::new(SynSugMirror {}),
            Box::new(Delta {}),
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
                let out_clone: Output = Output::clone(&*out);
                let Output {
                    expression,
                    spawn,
                    filter,
                    close,
                    ..
                } = out_clone;
                let (new_out_expr, cs) = self.desugarize_expression(expression.clone(), &ast, current_sugar);
                change_set += cs;
                let new_spawn_spec = if let Some(spawn_spec) = spawn {
                    let SpawnSpec {
                        target,
                        condition,
                        annotated_pacing,
                        is_if,
                        id,
                        span,
                    } = spawn_spec;
                    let target = target.map(|expr| {
                        let (new_expr, spawn_cs) = self.desugarize_expression(expr, &ast, current_sugar);
                        change_set += spawn_cs;
                        new_expr
                    });
                    let condition = condition.map(|expr| {
                        let (new_expr, spawn_cond_cs) = self.desugarize_expression(expr, &ast, current_sugar);
                        change_set += spawn_cond_cs;
                        new_expr
                    });
                    Some(SpawnSpec {
                        target,
                        condition,
                        annotated_pacing,
                        is_if,
                        id,
                        span,
                    })
                } else {
                    None
                };

                let new_filter_spec = if let Some(filter_spec) = filter {
                    let FilterSpec { target, id, span } = filter_spec;
                    let (new_target, filter_cs) = self.desugarize_expression(target, &ast, current_sugar);
                    change_set += filter_cs;
                    Some(FilterSpec {
                        target: new_target,
                        id,
                        span,
                    })
                } else {
                    None
                };

                let new_close_spec = if let Some(close_spec) = close {
                    let CloseSpec {
                        target,
                        id,
                        span,
                        annotated_pacing,
                    } = close_spec;
                    let (new_target, close_cs) = self.desugarize_expression(target, &ast, current_sugar);
                    change_set += close_cs;
                    Some(CloseSpec {
                        target: new_target,
                        id,
                        span,
                        annotated_pacing,
                    })
                } else {
                    None
                };

                let new_out = Output {
                    expression: new_out_expr,
                    spawn: new_spawn_spec,
                    filter: new_filter_spec,
                    close: new_close_spec,
                    ..out_clone
                };
                ast.outputs[ix] = Rc::new(new_out);
            }
            for ix in 0..ast.trigger.len() {
                let trigger = &ast.trigger[ix];
                let (new_out_expr, cs) = self.desugarize_expression(trigger.expression.clone(), &ast, current_sugar);
                change_set += cs;
                let trigger_clone: Trigger = Trigger::clone(&*trigger);
                let new_trigger = Trigger {
                    expression: new_out_expr,
                    ..trigger_clone
                };
                ast.trigger[ix] = Rc::new(new_trigger);
            }
            for input in ast.inputs.iter() {
                change_set += self.desugarize_input(input, &ast, current_sugar);
            }

            for output in ast.outputs.iter() {
                change_set += self.desugarize_output(output, &ast, current_sugar);
            }

            for trigger in ast.trigger.iter() {
                change_set += self.desugarize_trigger(trigger, &ast, current_sugar);
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
                    } else if let Some(idx) = ast.trigger.iter().position(|o| o.id == id) {
                        assert_eq!(Rc::strong_count(&ast.trigger[idx]), 1);
                        ast.trigger.remove(idx);
                    } else {
                        debug_assert!(false, "id in changeset does not belong to any stream");
                    }
                },
                ChangeInstruction::ReplaceExpr(id, expr) => {
                    for ix in 0..ast.outputs.len() {
                        let out = &ast.outputs[ix];
                        let out_clone: Output = Output::clone(&*out);
                        let Output {
                            expression,
                            spawn,
                            filter,
                            close,
                            ..
                        } = out_clone;
                        let new_out_expr = self.apply_expr_global_change(id, &expr, &expression, &ast);
                        let new_spawn_spec = if let Some(spawn_spec) = spawn {
                            let SpawnSpec {
                                target,
                                condition,
                                annotated_pacing,
                                is_if,
                                id,
                                span,
                            } = spawn_spec;
                            let target = target
                                .map(|tar_expression| self.apply_expr_global_change(id, &expr, &tar_expression, &ast));
                            let condition = condition.map(|cond_expression| {
                                self.apply_expr_global_change(id, &expr, &cond_expression, &ast)
                            });
                            Some(SpawnSpec {
                                target,
                                condition,
                                annotated_pacing,
                                is_if,
                                id,
                                span,
                            })
                        } else {
                            None
                        };

                        let new_filter_spec = if let Some(filter_spec) = filter {
                            let FilterSpec { target, id, span } = filter_spec;
                            let new_target = self.apply_expr_global_change(id, &expr, &target, &ast);
                            Some(FilterSpec {
                                target: new_target,
                                id,
                                span,
                            })
                        } else {
                            None
                        };

                        let new_close_spec = if let Some(close_spec) = close {
                            let CloseSpec {
                                target,
                                id,
                                span,
                                annotated_pacing,
                            } = close_spec;
                            let new_target = self.apply_expr_global_change(id, &expr, &target, &ast);
                            Some(CloseSpec {
                                target: new_target,
                                id,
                                span,
                                annotated_pacing,
                            })
                        } else {
                            None
                        };

                        let new_out = Output {
                            expression: new_out_expr,
                            spawn: new_spawn_spec,
                            filter: new_filter_spec,
                            close: new_close_spec,
                            ..out_clone
                        };
                        ast.outputs[ix] = Rc::new(new_out);
                    }

                    for ix in 0..ast.trigger.len() {
                        let trigger: &Rc<Trigger> = &ast.trigger[ix];
                        let new_trigger_expr = self.apply_expr_global_change(id, &expr, &trigger.expression, &ast);
                        let trigger_clone: Trigger = Trigger::clone(&*trigger);
                        let new_trigger = Trigger {
                            expression: new_trigger_expr,
                            ..trigger_clone
                        };
                        ast.trigger[ix] = Rc::new(new_trigger);
                    }
                },
            }
        });

        ast
    }

    fn apply_expr_global_change(
        &self,
        target_id: NodeId,
        new_expr: &Expression,
        ast_expr: &Expression,
        ast: &RtLolaAst,
    ) -> Expression {
        if ast_expr.id == target_id {
            return new_expr.clone();
        }
        let span = ast_expr.span.clone();

        use ExpressionKind::*;
        match &ast_expr.kind {
            Lit(_) | Ident(_) | MissingExpression => ast_expr.clone(),
            Unary(op, inner) => {
                Expression {
                    kind: Unary(
                        *op,
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Field(inner, ident) => {
                Expression {
                    kind: Field(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                        ident.clone(),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            StreamAccess(inner, acc_kind) => {
                Expression {
                    kind: StreamAccess(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                        *acc_kind,
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Offset(inner, offset) => {
                Expression {
                    kind: Offset(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
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
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
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
                        Box::new(self.apply_expr_global_change(target_id, new_expr, left, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, right, ast)),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Default(left, right) => {
                Expression {
                    kind: Default(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, left, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, right, ast)),
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
                        expr: Box::new(self.apply_expr_global_change(target_id, new_expr, left, ast)),
                        duration: Box::new(self.apply_expr_global_change(target_id, new_expr, right, ast)),
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
                        expr: Box::new(self.apply_expr_global_change(target_id, new_expr, left, ast)),
                        duration: Box::new(self.apply_expr_global_change(target_id, new_expr, right, ast)),
                        wait: *wait,
                        aggregation: *aggregation,
                    },
                    span,
                    ..*ast_expr
                }
            },
            Ite(condition, normal, alternative) => {
                Expression {
                    kind: Ite(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, condition, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, normal, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, alternative, ast)),
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
                            .map(|t_expr| self.apply_expr_global_change(target_id, new_expr, t_expr, ast))
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
                            .map(|t_expr| self.apply_expr_global_change(target_id, new_expr, t_expr, ast))
                            .collect(),
                    ),
                    span,
                    ..*ast_expr
                }
            },
            Method(base, name, types, arguments) => {
                Expression {
                    kind: Method(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, base, ast)),
                        name.clone(),
                        types.clone(),
                        arguments
                            .iter()
                            .map(|t_expr| self.apply_expr_global_change(target_id, new_expr, t_expr, ast))
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
        &self,
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
                let (inner, cs) = self.desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: Unary(op, Box::new(inner)),
                    span,
                    id,
                }
            },
            Field(inner, ident) => {
                let (inner, cs) = self.desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: Field(Box::new(inner), ident),
                    span,
                    id,
                }
            },
            StreamAccess(inner, acc_kind) => {
                let (inner, cs) = self.desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: StreamAccess(Box::new(inner), acc_kind),
                    span,
                    id,
                }
            },
            Offset(inner, offset) => {
                let (inner, cs) = self.desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: Offset(Box::new(inner), offset),
                    span,
                    id,
                }
            },
            ParenthesizedExpression(lp, inner, rp) => {
                let (inner, cs) = self.desugarize_expression(*inner, ast, current_sugar);
                return_cs += cs;
                Expression {
                    kind: ParenthesizedExpression(lp, Box::new(inner), rp),
                    span,
                    id,
                }
            },
            Binary(bin_op, left, right) => {
                let (left, lcs) = self.desugarize_expression(*left, ast, current_sugar);
                return_cs += lcs;
                let (right, rcs) = self.desugarize_expression(*right, ast, current_sugar);
                return_cs += rcs;
                Expression {
                    kind: Binary(bin_op, Box::new(left), Box::new(right)),
                    span,
                    id,
                }
            },
            Default(left, right) => {
                let (left, lcs) = self.desugarize_expression(*left, ast, current_sugar);
                return_cs += lcs;
                let (right, rcs) = self.desugarize_expression(*right, ast, current_sugar);
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
                let (expr, ecs) = self.desugarize_expression(*left, ast, current_sugar);
                return_cs += ecs;
                let (dur, dcs) = self.desugarize_expression(*right, ast, current_sugar);
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
                let (expr, ecs) = self.desugarize_expression(*left, ast, current_sugar);
                return_cs += ecs;
                let (dur, dcs) = self.desugarize_expression(*right, ast, current_sugar);
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
            Ite(condition, normal, alternative) => {
                let (condition, ccs) = self.desugarize_expression(*condition, ast, current_sugar);
                return_cs += ccs;
                let (normal, ncs) = self.desugarize_expression(*normal, ast, current_sugar);
                return_cs += ncs;
                let (alternative, acs) = self.desugarize_expression(*alternative, ast, current_sugar);
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
                    .map(|t_expr| self.desugarize_expression(t_expr, ast, current_sugar))
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
                    .map(|t_expr| self.desugarize_expression(t_expr, ast, current_sugar))
                    .unzip();
                return_cs += v_cs.into_iter().fold(ChangeSet::empty(), |acc, x| acc + x);
                Expression {
                    kind: Function(name, types, v_expr),
                    span,
                    id,
                }
            },
            Method(base, name, types, arguments) => {
                let (base_expr, ecs) = self.desugarize_expression(*base, ast, current_sugar);
                return_cs += ecs;
                let (v_expr, v_cs): (Vec<Expression>, Vec<ChangeSet>) = arguments
                    .into_iter()
                    .map(|t_expr| self.desugarize_expression(t_expr, ast, current_sugar))
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

    #[allow(clippy::borrowed_box)]
    fn desugarize_trigger(&self, trigger: &Trigger, ast: &RtLolaAst, current_sugar: &Box<dyn SynSugar>) -> ChangeSet {
        current_sugar.desugarize_stream_trigger(trigger, ast)
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
    fn test_aggr_replace() {
        let spec = "output x @ 5hz := x.count(6s)".to_string();
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        assert!(matches!(
            ast.outputs[0].expression.kind,
            ExpressionKind::SlidingWindowAggregation {
                aggregation: WindowOperation::Count,
                ..
            }
        ));
    }

    #[test]
    fn test_aggr_replace_nested() {
        let spec = "output x @ 5hz := -x.sum(6s)".to_string();
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].expression.kind.clone();
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
        let spec = "output x @ 5hz := x.avg(5s) - x.integral(2.5s)".to_string();
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].expression.kind.clone();
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
        let spec = "output x @ 5hz := x.last(3)".to_string();
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        let out_kind = ast.outputs[0].expression.kind.clone();
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
        let spec = "output y := delta(x)".to_string();
        let expected = "output y := x - x.offset(by: -1).defaults(to: 0)";
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        assert_eq!(expected, format!("{}", ast).trim());
    }

    #[test]
    fn test_mirror_replace() {
        let spec = "output x := 3 \noutput y mirrors x when x > 5".to_string();
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        assert_eq!(ast.outputs.len(), 2);
        assert!(ast.mirrors.is_empty());
        let new = &ast.outputs[1];
        let target = &ast.outputs[0];
        assert!(matches!(
            &target.name,
            Ident {
                name,
                ..
            }
            if name == &String::from("x")
        ));
        assert!(matches!(
            &new.name,
            Ident {
                name,
                ..
            }
            if name == &String::from("y")
        ));
        assert_eq!(new.annotated_type, target.annotated_type);
        assert_eq!(new.annotated_pacing_type, target.annotated_pacing_type);
        assert_eq!(new.close, target.close);
        assert_eq!(new.expression, target.expression);
        assert!(new.filter.is_some());
        assert!(matches!(
            new.filter.as_ref().unwrap().target,
            Expression {
                kind: ExpressionKind::Binary(..),
                ..
            }
        ));
        assert_eq!(new.params, target.params);
        assert_eq!(new.spawn, target.spawn);
    }
}
