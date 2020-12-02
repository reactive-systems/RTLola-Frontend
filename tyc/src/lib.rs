extern crate rtlola_frontend as front;
extern crate rusttyc;

mod pacing_ast_climber;
mod pacing_types;
mod rtltc;
mod value_ast_climber;
mod value_types;

use crate::rtltc::LolaTypeChecker;
use front::hir::modes::ir_expr::WithIrExpr;
use front::hir::modes::HirMode;
use front::hir::Hir;
use front::reporting::Handler;

pub use crate::rtltc::TypeTable;

pub fn type_check<M>(hir: &Hir<M>, handler: &Handler) -> Result<TypeTable, String>
where
    M: HirMode + WithIrExpr + 'static,
{
    let mut tyc = LolaTypeChecker::new(hir, handler);
    tyc.check()
}
