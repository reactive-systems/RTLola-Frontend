extern crate rusttyc;

pub(crate) mod pacing_ast_climber;
pub(crate) mod pacing_types;
pub(crate) mod rtltc;
pub(crate) mod value_ast_climber;
pub(crate) mod value_types;

use crate::hir::{modes::ir_expr::IrExprTrait, modes::HirMode, Hir};
use crate::reporting::Handler;
use crate::tyc::rtltc::LolaTypeChecker;

use crate::tyc::rtltc::TypeTable;

pub fn type_check<M>(hir: &Hir<M>, handler: &Handler) -> Result<TypeTable, String>
where
    M: HirMode + IrExprTrait + 'static,
{
    let mut tyc = LolaTypeChecker::new(hir, handler);
    tyc.check()
}
