use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use super::*;

#[derive(Debug, Clone)]
enum ChangeInstruction {
    ReplaceExpr(NodeId, Expression),
    #[allow(dead_code)] // currently unused
    ReplaceStream(NodeId, Box<Output>),
    #[allow(dead_code)] //currently unused
    Add(Box<Output>),
}

#[derive(Debug, Clone)]
struct ChangeSet {
    instructions: HashSet<ChangeInstruction>,
}

/// The container for all Syntactic Sugar structs.
///
/// Contains the logic on how and when to remove syntactic sugar changes.
/// A transformer just needs to implement the SynSugar trait and be registered in the internal vector, constructed in [Desugarizer::all].
#[allow(missing_debug_implementations)] //Contains no relevant fields, only logic. Trait object vectors are hard to print.
pub struct Desugarizer {
    sugar_transformers: Vec<Box<dyn SynSugar>>,
}

impl Desugarizer {
    /// Before calling this, all others Rcs on Ast nodes MUST be dropped.
    ///
    /// # panics
    /// if an Rc as a strong count greater then one.
    pub fn remove_syn_sugar(&self, ast: RtLolaAst) -> RtLolaAst {
        let mut change_set = ChangeSet::empty();

        //TODO we may need to apply chnages in order instead of collecting them first
        // Review needed on order of calls: streams or expressions first
        for out in ast.outputs.iter() {
            change_set += self.desugarize_expression(&out.expression, &ast);
        }
        for trigger in ast.trigger.iter() {
            change_set += self.desugarize_expression(&trigger.expression, &ast);
        }
        for input in ast.inputs.iter() {
            change_set += self.desugarize_input(input, &ast);
        }

        for output in ast.outputs.iter() {
            change_set += self.desugarize_output(output, &ast);
        }

        for trigger in ast.trigger.iter() {
            change_set += self.desugarize_trigger(trigger, &ast);
        }
        self.apply_change_set(change_set, ast)
    }

    fn apply_change_set(&self, c_s: ChangeSet, mut ast: RtLolaAst) -> RtLolaAst {
        dbg!(&c_s);
        c_s.into_iter().for_each(|ci| {
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
                        let new_out_expr = self.apply_expr_change(id, &expr, &out.expression, &ast);
                        let out_clone: Output = Output::clone(&*out);
                        let new_out = Output {
                            expression: new_out_expr,
                            ..out_clone
                        };
                        ast.outputs[ix] = Rc::new(new_out);
                    }

                    for ix in 0..ast.trigger.len() {
                        let out: &Rc<Trigger> = &ast.trigger[ix];
                        let new_out_expr = self.apply_expr_change(id, &expr, &out.expression, &ast);
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

    fn apply_expr_change(
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
                    kind: Unary(*op, Box::new(self.apply_expr_change(target_id, new_expr, inner, ast))),
                    ..ast_expr.clone()
                }
            },
            Field(inner, ident) => {
                Expression {
                    kind: Field(
                        Box::new(self.apply_expr_change(target_id, new_expr, inner, ast)),
                        ident.clone(),
                    ),
                    ..ast_expr.clone()
                }
            },
            StreamAccess(inner, acc_kind) => {
                Expression {
                    kind: StreamAccess(
                        Box::new(self.apply_expr_change(target_id, new_expr, inner, ast)),
                        *acc_kind,
                    ),
                    ..ast_expr.clone()
                }
            },
            Offset(inner, offset) => {
                Expression {
                    kind: Offset(
                        Box::new(self.apply_expr_change(target_id, new_expr, inner, ast)),
                        *offset,
                    ),
                    ..ast_expr.clone()
                }
            },
            ParenthesizedExpression(lp, inner, rp) => {
                Expression {
                    kind: ParenthesizedExpression(
                        lp.clone(),
                        Box::new(self.apply_expr_change(target_id, new_expr, inner, ast)),
                        rp.clone(),
                    ),
                    ..ast_expr.clone()
                }
            },
            Binary(binop, left, right) => {
                Expression {
                    kind: Binary(
                        *binop,
                        Box::new(self.apply_expr_change(target_id, new_expr, left, ast)),
                        Box::new(self.apply_expr_change(target_id, new_expr, right, ast)),
                    ),
                    ..ast_expr.clone()
                }
            },
            Default(left, right) => {
                Expression {
                    kind: Default(
                        Box::new(self.apply_expr_change(target_id, new_expr, left, ast)),
                        Box::new(self.apply_expr_change(target_id, new_expr, right, ast)),
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
                        expr: Box::new(self.apply_expr_change(target_id, new_expr, left, ast)),
                        duration: Box::new(self.apply_expr_change(target_id, new_expr, right, ast)),
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
                    kind: DiscreteWindowAggregation {
                        expr: Box::new(self.apply_expr_change(target_id, new_expr, left, ast)),
                        duration: Box::new(self.apply_expr_change(target_id, new_expr, right, ast)),
                        wait: *wait,
                        aggregation: *aggregation,
                    },
                    ..ast_expr.clone()
                }
            },
            Ite(cond, normal, alternative) => {
                Expression {
                    kind: Ite(
                        Box::new(self.apply_expr_change(target_id, new_expr, cond, ast)),
                        Box::new(self.apply_expr_change(target_id, new_expr, normal, ast)),
                        Box::new(self.apply_expr_change(target_id, new_expr, alternative, ast)),
                    ),
                    ..ast_expr.clone()
                }
            },
            Tuple(entries) => {
                Expression {
                    kind: Tuple(
                        entries
                            .iter()
                            .map(|t_expr| self.apply_expr_change(target_id, new_expr, t_expr, ast))
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
                            .map(|t_expr| self.apply_expr_change(target_id, new_expr, t_expr, ast))
                            .collect(),
                    ),
                    ..ast_expr.clone()
                }
            },
            Method(base, name, types, arguments) => {
                Expression {
                    kind: Method(
                        Box::new(self.apply_expr_change(target_id, new_expr, base, ast)),
                        name.clone(),
                        types.clone(),
                        arguments
                            .iter()
                            .map(|t_expr| self.apply_expr_change(target_id, new_expr, t_expr, ast))
                            .collect(),
                    ),
                    ..ast_expr.clone()
                }
            },
        }
    }

    /// Constructs a Desugarsizer with all registered SynSugar transformations.
    ///
    /// All transformations registered in the internal vector will be applied on the ast.
    /// New additions need to be added in this function.
    pub fn all() -> Self {
        let all_transformers: Vec<Box<dyn SynSugar>> = vec![Box::new(AggrMethodToWindow {})];
        Self {
            sugar_transformers: all_transformers,
        }
    }

    fn desugarize_expression(&self, expr: &Expression, ast: &RtLolaAst) -> ChangeSet {
        use ExpressionKind::*;
        let mut change_set = ChangeSet::empty();
        for sugar in self.sugar_transformers.iter() {
            match &expr.kind.clone() {
                Lit(_) | Ident(_) | MissingExpression => {
                    change_set += sugar.desugarize_expr(expr, ast);
                },
                Unary(_, inner)
                | Field(inner, _)
                | StreamAccess(inner, _)
                | Offset(inner, _)
                | ParenthesizedExpression(_, inner, _) => {
                    change_set += self.desugarize_expression(inner, ast);
                    change_set += sugar.desugarize_expr(expr, ast);
                },
                Binary(_, left, right)
                | Default(left, right)
                | DiscreteWindowAggregation {
                    expr: left,
                    duration: right,
                    ..
                }
                | SlidingWindowAggregation {
                    expr: left,
                    duration: right,
                    ..
                } => {
                    change_set += self.desugarize_expression(left, ast);
                    change_set += self.desugarize_expression(right, ast);
                    change_set += sugar.desugarize_expr(expr, ast);
                },
                Ite(cond, normal, alternative) => {
                    change_set += self.desugarize_expression(cond, ast);
                    change_set += self.desugarize_expression(normal, ast);
                    change_set += self.desugarize_expression(alternative, ast);
                    change_set += sugar.desugarize_expr(expr, ast);
                },
                Tuple(entries) | Function(_, _, entries) => {
                    change_set += entries
                        .iter()
                        .map(|inner_expr| self.desugarize_expression(inner_expr, ast))
                        .fold(ChangeSet::empty(), |acc, v| acc + v);
                    change_set += sugar.desugarize_expr(expr, ast);
                },
                Method(base, _, _, arguments) => {
                    change_set += arguments
                        .iter()
                        .map(|inner_expr| self.desugarize_expression(inner_expr, ast))
                        .fold(ChangeSet::empty(), |acc, v| acc + v);
                    change_set += sugar.desugarize_expr(base, ast);
                    change_set += sugar.desugarize_expr(expr, ast);
                },
            }
        }
        change_set
    }

    fn desugarize_input(&self, input: &Input, ast: &RtLolaAst) -> ChangeSet {
        let mut change_set = ChangeSet::empty();
        for sugar in self.sugar_transformers.iter() {
            change_set += sugar.desugarize_stream_in(input, ast);
        }
        change_set
    }

    fn desugarize_output(&self, input: &Output, ast: &RtLolaAst) -> ChangeSet {
        let mut change_set = ChangeSet::empty();
        for sugar in self.sugar_transformers.iter() {
            change_set += sugar.desugarize_stream_out(input, ast);
        }
        change_set
    }

    fn desugarize_trigger(&self, input: &Trigger, ast: &RtLolaAst) -> ChangeSet {
        let mut change_set = ChangeSet::empty();
        for sugar in self.sugar_transformers.iter() {
            change_set += sugar.desugarize_stream_trigger(input, ast);
        }
        change_set
    }
}

impl Default for Desugarizer {
    fn default() -> Self {
        Self::all()
    }
}

#[allow(unused_variables)] //allow unsued arguments in the default implementation without changing the name
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

//changeset from iterator
//changeset single 1 element

#[derive(Debug)]
struct AggrMethodToWindow {}

impl AggrMethodToWindow {
    fn apply(&self, expr: &Expression) -> ChangeSet {
        match &expr.kind {
            ExpressionKind::Method(base, name, _types, arguments) => {
                let op = match name.name.name.as_ref() {
                    "count" => WindowOperation::Count,
                    "min" => WindowOperation::Min,
                    "max" => WindowOperation::Max,
                    "sum" => WindowOperation::Sum,
                    "avg" => WindowOperation::Average,
                    "integral" => WindowOperation::Integral,
                    /*
                    "var" => WindowOperation::Variance,
                    "cov" => WindowOperation::Covariance,
                    "sd" => WindowOperation::StandardDeviation,
                    "med" => WindowOperation::NthPercentile(50),
                    */
                    _ => return ChangeSet::empty(),
                };
                let target_stream = base.clone();
                let wait = false;
                let duration = Box::new(arguments[0].clone());
                let mut new_id = expr.id;
                new_id.prime_counter += 1;
                let new_expr = Expression {
                    kind: ExpressionKind::SlidingWindowAggregation {
                        expr: target_stream,
                        duration,
                        wait,
                        aggregation: op,
                    },
                    id: new_id,
                    span: expr.span.clone(),
                };
                ChangeSet::single(ChangeInstruction::ReplaceExpr(expr.id, new_expr))
            },
            _ => ChangeSet::empty(), //Do nothing
        }
    }
}

impl SynSugar for AggrMethodToWindow {
    fn desugarize_expr<'a>(&self, exp: &'a Expression, _ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(exp)
    }
}

impl ChangeSet {
    fn empty() -> Self {
        Self {
            instructions: HashSet::new(),
        }
    }

    fn single(instr: ChangeInstruction) -> Self {
        Self {
            instructions: std::iter::once(instr).collect(),
        }
    }

    /* currently unused
    fn from(itr: impl IntoIterator<Item = ChangeInstruction>) -> Self {
        Self {
            instructions: itr.into_iter().collect(),
        }
    }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &ChangeInstruction> + 'a> {
        Box::new(self.instructions.iter())
    }
    */

    fn into_iter<'a>(self) -> Box<dyn Iterator<Item = ChangeInstruction> + 'a> {
        Box::new(self.instructions.into_iter())
    }
}

impl std::ops::Add<ChangeSet> for ChangeSet {
    type Output = ChangeSet;

    fn add(self, rhs: Self) -> Self {
        let set = self.instructions.union(&rhs.instructions).cloned().collect();
        Self { instructions: set }
    }
}

impl std::ops::AddAssign<ChangeSet> for ChangeSet {
    fn add_assign(&mut self, rhs: Self) {
        let set = self.instructions.union(&rhs.instructions).cloned().collect();
        *self = Self { instructions: set }
    }
}

impl std::ops::Add<ChangeInstruction> for ChangeSet {
    type Output = ChangeSet;

    fn add(mut self, rhs: ChangeInstruction) -> Self {
        self.instructions.insert(rhs);
        self
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

    #[test]
    fn test_aggr_replace() {
        let spec = "output x @ 5hz := x.count(6s)".to_string();
        let ast = crate::parse(crate::ParserConfig::for_string(spec)).unwrap();
        dbg!(&ast);
        let sugar = Desugarizer::all();
        let new_ast: RtLolaAst = sugar.remove_syn_sugar(ast);
        dbg!(&new_ast);
        assert!(matches!(
            new_ast.outputs[0].expression.kind,
            ExpressionKind::SlidingWindowAggregation {
                aggregation: WindowOperation::Count,
                ..
            }
        ));
    }
}
