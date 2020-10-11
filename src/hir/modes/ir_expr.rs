use crate::{
    common_ir::SRef,
    hir::{expression::Expression, Hir, Window},
};

use super::IrExpression;

pub(crate) trait WithIrExpr {
    fn windows(&self) -> Vec<Window>;
    fn expr(&self, sr: SRef) -> &Expression;
}

impl WithIrExpr for IrExpression {
    fn windows(&self) -> Vec<Window> {
        self.windows.clone()
    }
    fn expr(&self, sr: SRef) -> &Expression {
        self.expressions.get(&sr).expect("accessing non-existent expression")
    }
}

impl Hir<IrExpression> {
    fn windows(&self) -> Vec<Window> {
        self.mode.windows.clone()
    }
}

pub(crate) trait IrExprWrapper {
    type InnerE: WithIrExpr;
    fn inner_expr(&self) -> &Self::InnerE;
}

impl<A: IrExprWrapper<InnerE = T>, T: WithIrExpr + 'static> WithIrExpr for A {
    fn windows(&self) -> Vec<Window> {
        self.inner_expr().windows()
    }
    fn expr(&self, sr: SRef) -> &Expression {
        self.inner_expr().expr(sr)
    }
}
