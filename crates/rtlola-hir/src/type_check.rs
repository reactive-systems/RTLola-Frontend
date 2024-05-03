mod pacing_ast_climber;
mod pacing_types;
mod rtltc;
mod value_ast_climber;
mod value_types;

use rtlola_reporting::RtLolaError;
use uom::si::rational64::Frequency as UOM_Frequency;

pub use self::pacing_types::ActivationCondition;
use crate::hir::{Expression, Hir};
use crate::modes::{HirMode, Typed};
use crate::type_check::rtltc::LolaTypeChecker;

/// Checks all types of in the [Hir] and returns a [Typed] struct, containing all type information.
/// In case of a type error a string with sparse error description is returned and the [Handler] emits
/// detailed error information.
pub(crate) fn type_check<M>(hir: &Hir<M>) -> Result<Typed, RtLolaError>
where
    M: HirMode + 'static,
{
    let mut tyc = LolaTypeChecker::new(hir);
    tyc.check()
}

/// The external definition of a pacing type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcretePacingType {
    /// The stream / expression can be evaluated whenever the activation condition is satisfied.
    Event(ActivationCondition),
    /// The stream / expression can be evaluated with a fixed global frequency.
    FixedGlobalPeriodic(UOM_Frequency),
    /// The stream / expression can be evaluated with a fixed local frequency.
    FixedLocalPeriodic(UOM_Frequency),
    /// The stream / expression can be evaluated with any fixed frequency.
    FixedAnyPeriodic(UOM_Frequency),
    /// The stream / expression can be evaluated with any global frequency.
    GlobalPeriodic,
    /// The stream / expression can be evaluated with any local frequency.
    LocalPeriodic,
    /// The stream / expression can be evaluated with any frequency.
    AnyPeriodic,
    /// The stream / expression can always be evaluated.
    Constant,
}

impl ConcretePacingType {
    /// Returns true if the type is fixed-periodic
    pub fn is_periodic(&self) -> bool {
        matches!(
            self,
            ConcretePacingType::FixedGlobalPeriodic(_) | ConcretePacingType::FixedLocalPeriodic(_)
        )
    }

    /// Returns true if the type is event-based
    pub fn is_event_based(&self) -> bool {
        matches!(self, ConcretePacingType::Event(_))
    }

    /// Returns true if the type is constant
    pub fn is_constant(&self) -> bool {
        matches!(self, ConcretePacingType::Constant)
    }
}

/// The external definition for a value type.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ConcreteValueType {
    /// Bool e.g. true, false
    Bool,
    /// 8-bit signed integer
    Integer8,
    /// 16-bit signed integer
    Integer16,
    /// 32-bit signed integer
    Integer32,
    /// 64-bit signed integer
    Integer64,
    /// 8-bit unsigned integer
    UInteger8,
    /// 16-bit unsigned integer
    UInteger16,
    /// 32-bit unsigned integer
    UInteger32,
    /// 64-bit unsigned integer
    UInteger64,
    /// 32-bit floating point value
    Float32,
    /// 64-bit floating point value
    Float64,
    /// A tuple type of arbitrary but fixed length: (Int8, Float32, Bool)
    Tuple(Vec<ConcreteValueType>),
    /// String value: "Hello"
    TString,
    /// Byte value, used for string index access and regex matches
    Byte,
    /// Optional value for partial functions like [Offset](crate::hir::Offset)
    Option(Box<ConcreteValueType>),
}

/// The external definition of the stream pacing.
#[derive(Debug, Clone)]
pub struct ConcreteStreamPacing {
    /// The pacing of the stream expression.
    pub eval_pacing: ConcretePacingType,
    /// The filter expression
    pub eval_condition: Expression,
    /// The pacing of the spawn expression
    pub spawn_pacing: ConcretePacingType,
    /// The spawn condition expression
    pub spawn_condition: Expression,
    /// The pacing of the close expression.
    pub close_pacing: ConcretePacingType,
    /// The close expression
    pub close_condition: Expression,
}

/// The external definition of the stream type.
#[derive(Debug, Clone)]
pub struct StreamType {
    /// The [ConcreteValueType] of the stream and his expression, e.g. Bool.
    pub value_ty: ConcreteValueType,
    /// The [ConcretePacingType] of the stream, e.g. 5Hz.
    pub eval_pacing: ConcretePacingType,
    /// The filter type given by the filter expression.
    /// The stream only has to be evaluated if this boolean expression evaluates to true.
    pub eval_condition: Expression,
    /// The pacing of the spawn expression.
    pub spawn_pacing: ConcretePacingType,
    /// The spawn condition of the stream. The spawn expression only has to be evaluated if this expression evaluates to true.
    pub spawn_condition: Expression,
    /// The pacing of the close condition.
    pub close_pacing: ConcretePacingType,
    /// The stream can be closed and does not have to be evaluated if this boolean expression returns true.
    pub close_condition: Expression,
}
