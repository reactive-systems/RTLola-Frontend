extern crate rusttyc;

pub(crate) mod pacing_ast_climber;
pub(crate) mod pacing_types;
pub(crate) mod rtltc;
pub(crate) mod value_ast_climber;
pub(crate) mod value_types;

use crate::hir::{modes::HirMode, modes::IrExprTrait, modes::Typed, Hir};
use crate::reporting::Handler;
use crate::tyc::rtltc::LolaTypeChecker;

pub fn type_check<M>(hir: &Hir<M>, handler: &Handler) -> Result<Typed, String>
where
    M: HirMode + IrExprTrait + 'static,
{
    let mut tyc = LolaTypeChecker::new(hir, handler);
    tyc.check()
}
