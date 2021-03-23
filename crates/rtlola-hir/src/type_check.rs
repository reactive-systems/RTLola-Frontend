mod pacing_ast_climber;
mod pacing_types;
mod rtltc;
mod value_ast_climber;
mod value_types;

use self::pacing_types::ActivationCondition;
use crate::hir::{Expression, Hir};
use crate::type_check::rtltc::LolaTypeChecker;
use crate::{modes::HirMode, modes::IrExprTrait, modes::Typed};
use rtlola_reporting::Handler;
use uom::si::rational64::Frequency as UOM_Frequency;

pub(crate) fn type_check<M>(hir: &Hir<M>, handler: &Handler) -> Result<Typed, String>
where
    M: HirMode + IrExprTrait + 'static,
{
    let mut tyc = LolaTypeChecker::new(hir, handler);
    tyc.check()
}

/// The external definition of a pacing type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcretePacingType {
    /// The stream / expression can be evaluated whenever the activation condition is satisfied.
    Event(ActivationCondition),
    /// The stream / expression can be evaluated with a fixed frequency.
    FixedPeriodic(UOM_Frequency),
    /// The stream / expression can be evaluated with any frequency.
    Periodic,
    /// The stream / expression can always be evaluated.
    Constant,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ConcreteValueType {
    Bool,
    Integer8,
    Integer16,
    Integer32,
    Integer64,
    UInteger8,
    UInteger16,
    UInteger32,
    UInteger64,
    Float32,
    Float64,
    Tuple(Vec<ConcreteValueType>),
    TString,
    Byte,
    Option(Box<ConcreteValueType>),
}

/// The external definition of the stream pacing
#[derive(Debug, Clone)]
pub struct ConcreteStreamPacing {
    /// The pacing of the stream expression
    pub expression_pacing: ConcretePacingType,
    /// First element is the pacing of the spawn expression
    /// Second element is the spawn condition expression
    pub spawn: (ConcretePacingType, Expression),
    /// The filter expression
    pub filter: Expression,
    /// The close expression
    pub close: Expression,
}

#[derive(Debug, Clone)]
pub struct StreamType {
    pub value_ty: ConcreteValueType,
    pub pacing_ty: ConcretePacingType,
    pub spawn: (ConcretePacingType, Expression),
    pub filter: Expression,
    pub close: Expression,
}
