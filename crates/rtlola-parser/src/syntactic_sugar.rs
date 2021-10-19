use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

// List for all syntactic sugar transformer
mod aggregation_method;
use aggregation_method::AggrMethodToWindow;

use crate::ast::{Expression, ExpressionKind, Input, NodeId, Output, RtLolaAst, Trigger};

#[derive(Debug, Clone)]
enum ChangeInstruction {
    #[allow(dead_code)] // currently unused
    ReplaceExpr(NodeId, Expression),
    #[allow(dead_code)] // currently unused
    ReplaceStream(NodeId, Box<Output>),
    #[allow(dead_code)] //currently unused
    Add(Box<Output>),
}
#[derive(Debug, Clone)]
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

/// The container for all syntactic sugar structs.
///
/// Contains the logic on how and when to remove syntactic sugar changes.
/// A transformer just needs to implement the SynSugar trait and be registered in the internal vector, constructed in [Desugarizer::all].
#[allow(missing_debug_implementations)] //Contains no relevant fields, only logic. Trait object vectors are hard to print.
pub struct Desugarizer {
    sugar_transformers: Vec<Box<dyn SynSugar>>,
}

impl Desugarizer {
    /// Constructs a Desugarizer with all registered SynSugar transformations.
    ///
    /// All transformations registered in the internal vector will be applied on the ast.
    /// New structs have to be added in this function.
    pub fn all() -> Self {
        let all_transformers: Vec<Box<dyn SynSugar>> = vec![Box::new(AggrMethodToWindow {})];
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
            //magic rust do-while loop
            let (new_ast, change_flag) = self.desugarize_fp(ast);
            ast = new_ast;
            change_flag
        } {} //do not remove! magic rust do-while loop
        ast
    }

    fn desugarize_fp(&self, mut ast: RtLolaAst) -> (RtLolaAst, bool) {
        let mut change_flag = false;
        for current_sugar in self.sugar_transformers.iter() {
            let mut change_set = ChangeSet::empty();

            for ix in 0..ast.outputs.len() {
                let out = &ast.outputs[ix];
                let (new_out_expr, cs) = self.desugarize_expression(out.expression.clone(), &ast, current_sugar);
                change_set += cs;
                let out_clone: Output = Output::clone(&*out);
                let new_out = Output {
                    expression: new_out_expr,
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
                ChangeInstruction::Add(o) => {
                    ast.outputs.push(Rc::new(*o));
                },
                ChangeInstruction::ReplaceStream(id, out) => {
                    let idx = ast.outputs.iter().position(|o| o.id == id).unwrap();
                    assert_eq!(Rc::strong_count(&ast.outputs[idx]), 1);
                    ast.outputs[idx] = Rc::new(*out);
                },
                ChangeInstruction::ReplaceExpr(id, expr) => {
                    for ix in 0..ast.outputs.len() {
                        let out: &Rc<Output> = &ast.outputs[ix];
                        let new_out_expr = self.apply_expr_global_change(id, &expr, &out.expression, &ast);
                        let out_clone: Output = Output::clone(&*out);
                        let new_out = Output {
                            expression: new_out_expr,
                            ..out_clone
                        };
                        ast.outputs[ix] = Rc::new(new_out);
                    }

                    for ix in 0..ast.trigger.len() {
                        let out: &Rc<Trigger> = &ast.trigger[ix];
                        let new_out_expr = self.apply_expr_global_change(id, &expr, &out.expression, &ast);
                        let out_clone: Trigger = Trigger::clone(&*out);
                        let new_out = Trigger {
                            expression: new_out_expr,
                            ..out_clone
                        };
                        ast.trigger[ix] = Rc::new(new_out);
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

        use ExpressionKind::*;
        match &ast_expr.kind {
            Lit(_) | Ident(_) | MissingExpression => ast_expr.clone(),
            Unary(op, inner) => {
                Expression {
                    kind: Unary(
                        *op,
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                    ),
                    ..ast_expr.clone()
                }
            },
            Field(inner, ident) => {
                Expression {
                    kind: Field(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                        ident.clone(),
                    ),
                    ..ast_expr.clone()
                }
            },
            StreamAccess(inner, acc_kind) => {
                Expression {
                    kind: StreamAccess(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                        *acc_kind,
                    ),
                    ..ast_expr.clone()
                }
            },
            Offset(inner, offset) => {
                Expression {
                    kind: Offset(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                        *offset,
                    ),
                    ..ast_expr.clone()
                }
            },
            ParenthesizedExpression(lp, inner, rp) => {
                Expression {
                    kind: ParenthesizedExpression(
                        lp.clone(),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, inner, ast)),
                        rp.clone(),
                    ),
                    ..ast_expr.clone()
                }
            },
            Binary(bin_op, left, right) => {
                Expression {
                    kind: Binary(
                        *bin_op,
                        Box::new(self.apply_expr_global_change(target_id, new_expr, left, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, right, ast)),
                    ),
                    ..ast_expr.clone()
                }
            },
            Default(left, right) => {
                Expression {
                    kind: Default(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, left, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, right, ast)),
                    ),
                    ..ast_expr.clone()
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
                    ..ast_expr.clone()
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
                    ..ast_expr.clone()
                }
            },
            Ite(condition, normal, alternative) => {
                Expression {
                    kind: Ite(
                        Box::new(self.apply_expr_global_change(target_id, new_expr, condition, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, normal, ast)),
                        Box::new(self.apply_expr_global_change(target_id, new_expr, alternative, ast)),
                    ),
                    ..ast_expr.clone()
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
                    ..ast_expr.clone()
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
                    ..ast_expr.clone()
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
                    ..ast_expr.clone()
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

#[allow(unused_variables)] //allow unused arguments in the default implementation without changing the name
trait SynSugar {
    fn desugarize_expr<'a>(&self, exp: &'a Expression, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }
    fn desugarize_stream_out<'a>(&self, stream: &'a Output, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }
    fn desugarize_stream_in<'a>(&self, stream: &'a Input, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }
    fn desugarize_stream_trigger<'a>(&self, stream: &'a Trigger, ast: &'a RtLolaAst) -> ChangeSet {
        ChangeSet::empty()
    }
}

impl ChangeSet {
    fn empty() -> Self {
        Self {
            _local_applied_flag: false,
            local_instructions: None,
            global_instructions: HashSet::new(),
        }
    }

    fn single_local(instr: LocalChangeInstruction) -> Self {
        Self {
            _local_applied_flag: false,
            local_instructions: Some(instr),
            global_instructions: HashSet::new(),
        }
    }

    fn global_iter<'a>(self) -> Box<dyn Iterator<Item = ChangeInstruction> + 'a> {
        Box::new(self.global_instructions.into_iter())
    }

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

impl Hash for ChangeInstruction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        //TODO maybe add prime_counter into hash function
        match self {
            ChangeInstruction::ReplaceExpr(id, exp) => {
                id.id.hash(state);
                exp.id.id.hash(state);
            },
            ChangeInstruction::ReplaceStream(id, exp) => {
                id.id.hash(state);
                exp.id.id.hash(state);
            },
            ChangeInstruction::Add(out) => out.id.id.hash(state),
        }
    }
}

impl PartialEq for ChangeInstruction {
    fn eq(&self, rhs: &Self) -> bool {
        use ChangeInstruction::*;
        match (self, rhs) {
            (ReplaceExpr(a, b), ReplaceExpr(x, y)) => a == x && b.id == y.id,
            (ReplaceStream(a, b), ReplaceStream(x, y)) => a == x && b.id == y.id,
            (Add(o), Add(o2)) => o.id == o2.id,
            (_, _) => false,
        }
    }
}
impl Eq for ChangeInstruction {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, UnOp, WindowOperation};

    #[test]
    fn test_aggr_replace() {
        let spec = "output x @ 5hz := x.count(6s)".to_string();
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        let sugar = Desugarizer::all();
        let new_ast: RtLolaAst = sugar.remove_syn_sugar(ast);
        assert!(matches!(
            new_ast.outputs[0].expression.kind,
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
        let sugar = Desugarizer::all();
        let new_ast: RtLolaAst = sugar.remove_syn_sugar(ast);
        let out_kind = new_ast.outputs[0].expression.kind.clone();
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
        let sugar = Desugarizer::all();
        let new_ast: RtLolaAst = sugar.remove_syn_sugar(ast);
        let out_kind = new_ast.outputs[0].expression.kind.clone();
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
}
