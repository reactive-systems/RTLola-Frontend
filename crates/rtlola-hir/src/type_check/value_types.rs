use std::cmp::max;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use itertools::Itertools;
use rtlola_reporting::{Diagnostic, Span};
use rusttyc::{Arity, Constructable, Partial, TcErr, TcKey, Variant};

use super::*;
use crate::hir::{AnnotatedType, StreamReference};
use crate::type_check::rtltc::{Resolvable, TypeError};

/// The error kind for all custom errors during the value type check.
#[derive(Debug, Clone)]
pub(crate) enum ValueErrorKind {
    /// two conflicting types
    TypeClash(AbstractValueType, AbstractValueType),
    /// an error for tuple access/size
    TupleSize(usize, usize),
    /// exceeding the upper type bound
    ReificationTooWide(AbstractValueType),
    /// type not constrained enough
    CannotReify(AbstractValueType),
    /// annotated type exceeds the upper bound
    AnnotationTooWide(AnnotatedType),
    /// type not allowed as annotation
    AnnotationInvalid(AnnotatedType),
    /// Inferred, Expected
    ExactTypeMismatch(ConcreteValueType, ConcreteValueType),
    /// invalid child access for given type
    AccessOutOfBound(AbstractValueType, usize),
    /// Type, inferred, reported
    ArityMismatch(AbstractValueType, usize, usize),
    /// Child Construction Error, Parent Type, Index of Child
    ChildConstruction(Box<Self>, AbstractValueType, usize),
    /// A function call has more type parameters then needed or expected
    UnnecessaryTypeParam(Span),
    /// Inner expression of widen is wider than target width
    InvalidWiden(ConcreteValueType, ConcreteValueType),
    /// Optional Type is not allowed
    OptionNotAllowed(ConcreteValueType),
    /// The message of a trigger is not of type string
    WrongTriggerMsg(ConcreteValueType),
}

/// The [AbstractValueType] is used during inference and represents a value within the type lattice
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum AbstractValueType {
    Any,
    /// A numeric value which is either an integer or a float
    Numeric,
    SignedNumeric,
    /// Either a signed or and unsigned integer
    Integer,
    /// An signed integer of arbitrary size
    SInteger,
    SizedSInteger(u32),
    /// An unsigned integer of arbitrary size
    UInteger,
    SizedUInteger(u32),
    /// A float of arbitrary size
    Float,
    SizedFloat(u32),
    Bool,
    AnyTuple,
    Tuple(usize),
    Sequence,
    String,
    Bytes,
    Option,
}

impl Variant for AbstractValueType {
    type Err = ValueErrorKind;

    fn top() -> Self {
        Self::Any
    }

    fn meet(
        lhs: Partial<AbstractValueType>,
        rhs: Partial<AbstractValueType>,
    ) -> Result<Partial<AbstractValueType>, Self::Err> {
        use ValueErrorKind::*;
        fn tuple_meet(least_arity: usize, size: usize) -> Result<(AbstractValueType, usize), ValueErrorKind> {
            if least_arity <= size {
                Ok((Tuple(size), size))
            } else {
                Err(TupleSize(least_arity, size))
            }
        }
        use AbstractValueType::*;
        let (new_var, min_arity) = match (lhs.variant, rhs.variant) {
            (Any, other) => Ok((other, rhs.least_arity)),
            (other, Any) => Ok((other, lhs.least_arity)),
            (Numeric, Numeric) => Ok((Numeric, 0)),
            (SignedNumeric, SignedNumeric) => Ok((SignedNumeric, 0)),
            (Integer, Integer) => Ok((Integer, 0)),
            (SInteger, SInteger) => Ok((SInteger, 0)),
            (UInteger, UInteger) => Ok((UInteger, 0)),
            (Float, Float) => Ok((Float, 0)),
            (SInteger, SizedSInteger(x)) | (SizedSInteger(x), SInteger) => Ok((SizedSInteger(x), 0)),
            (SizedSInteger(l), SizedSInteger(r)) if l == r => Ok((SizedSInteger(l), 0)),
            (SizedSInteger(_), SizedSInteger(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (UInteger, SizedUInteger(x)) | (SizedUInteger(x), UInteger) => Ok((SizedUInteger(x), 0)),
            (SizedUInteger(l), SizedUInteger(r)) if l == r => Ok((SizedUInteger(l), 0)),
            (SizedUInteger(_), SizedUInteger(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Float, SizedFloat(x)) | (SizedFloat(x), Float) => Ok((SizedFloat(x), 0)),
            (SizedFloat(l), SizedFloat(r)) if l == r => Ok((SizedFloat(l), 0)),
            (SizedFloat(_), SizedFloat(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Bool, Bool) => Ok((Bool, 0)),
            (Bool, _) | (_, Bool) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Numeric, Integer) | (Integer, Numeric) => Ok((Integer, 0)),
            (Numeric, SInteger) | (SInteger, Numeric) => Ok((SInteger, 0)),
            (Numeric, SizedSInteger(w)) | (SizedSInteger(w), Numeric) => Ok((SizedSInteger(w), 0)),
            (Numeric, UInteger) | (UInteger, Numeric) => Ok((UInteger, 0)),
            (Numeric, SizedUInteger(w)) | (SizedUInteger(w), Numeric) => Ok((SizedUInteger(w), 0)),
            (Numeric, Float) | (Float, Numeric) => Ok((Float, 0)),
            (Numeric, SizedFloat(i)) | (SizedFloat(i), Numeric) => Ok((SizedFloat(i), 0)),
            (Numeric, SignedNumeric) | (SignedNumeric, Numeric) => Ok((SignedNumeric, 0)),
            (SignedNumeric, SInteger) | (SInteger, SignedNumeric) => Ok((SInteger, 0)),
            (SignedNumeric, SizedSInteger(w)) | (SizedSInteger(w), SignedNumeric) => Ok((SizedSInteger(w), 0)),
            (SignedNumeric, Float) | (Float, SignedNumeric) => Ok((Float, 0)),
            (SignedNumeric, SizedFloat(w)) | (SizedFloat(w), SignedNumeric) => Ok((SizedFloat(w), 0)),
            (Integer, SInteger) | (SInteger, Integer) => Ok((SInteger, 0)),
            (Integer, UInteger) | (UInteger, Integer) => Ok((UInteger, 0)),
            (Integer, SizedSInteger(x)) | (SizedSInteger(x), Integer) => Ok((SizedSInteger(x), 0)),
            (Integer, SizedUInteger(x)) | (SizedUInteger(x), Integer) => Ok((SizedUInteger(x), 0)),
            (Integer, _) | (_, Integer) => Err(TypeClash(lhs.variant, rhs.variant)),
            (SInteger, _) | (_, SInteger) => Err(TypeClash(lhs.variant, rhs.variant)),
            (SizedSInteger(_), _) | (_, SizedSInteger(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (UInteger, _) | (_, UInteger) => Err(TypeClash(lhs.variant, rhs.variant)),
            (SizedUInteger(_), _) | (_, SizedUInteger(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (AnyTuple, AnyTuple) => Ok((AnyTuple, max(lhs.least_arity, rhs.least_arity))),
            (AnyTuple, Tuple(size)) => tuple_meet(lhs.least_arity, size),
            (Tuple(size), AnyTuple) => tuple_meet(rhs.least_arity, size),
            (AnyTuple, _) | (_, AnyTuple) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Tuple(size_l), Tuple(size_r)) if size_l == size_r => Ok((Tuple(size_l), size_l)),
            (Tuple(size_l), Tuple(size_r)) => Err(TupleSize(size_l, size_r)),
            (Tuple(_), _) | (_, Tuple(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Sequence, String) | (String, Sequence) => Ok((String, 0)),
            (Sequence, Bytes) | (Bytes, Sequence) => Ok((Bytes, 0)),
            (SignedNumeric, _) | (_, SignedNumeric) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Sequence, _) | (_, Sequence) => Err(TypeClash(lhs.variant, rhs.variant)),
            (String, String) => Ok((String, 0)),
            (String, _) | (_, String) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Bytes, Bytes) => Ok((Bytes, 0)),
            (Bytes, _) | (_, Bytes) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Option, Option) => Ok((Option, 1)),
            (Option, _) | (_, Option) => Err(TypeClash(lhs.variant, rhs.variant)),
        }?;
        Ok(Partial {
            variant: new_var,
            least_arity: min_arity,
        })
    }

    fn arity(&self) -> Arity {
        use AbstractValueType::*;
        match self {
            Any | AnyTuple => Arity::Variable,
            Tuple(x) => Arity::Fixed(*x),
            Option => Arity::Fixed(1),
            Numeric | SignedNumeric | Integer | SInteger | SizedSInteger(_) | UInteger | SizedUInteger(_) | Float
            | SizedFloat(_) | Bool | Sequence | String | Bytes => Arity::Fixed(0),
        }
    }
}

impl Constructable for AbstractValueType {
    type Type = ConcreteValueType;

    fn construct(&self, children: &[ConcreteValueType]) -> Result<ConcreteValueType, ValueErrorKind> {
        use ValueErrorKind::*;
        match self {
            AbstractValueType::Any => Err(CannotReify(*self)),
            AbstractValueType::AnyTuple => Err(CannotReify(*self)),
            AbstractValueType::SInteger => Ok(ConcreteValueType::Integer64),
            AbstractValueType::SizedSInteger(w) if *w <= 8 => Ok(ConcreteValueType::Integer8),
            AbstractValueType::SizedSInteger(w) if *w <= 16 => Ok(ConcreteValueType::Integer16),
            AbstractValueType::SizedSInteger(w) if *w <= 32 => Ok(ConcreteValueType::Integer32),
            AbstractValueType::SizedSInteger(w) if *w <= 64 => Ok(ConcreteValueType::Integer64),
            AbstractValueType::SizedSInteger(_) => Err(ReificationTooWide(*self)),
            AbstractValueType::UInteger => Ok(ConcreteValueType::UInteger64),
            AbstractValueType::SizedUInteger(w) if *w <= 8 => Ok(ConcreteValueType::UInteger8),
            AbstractValueType::SizedUInteger(w) if *w <= 16 => Ok(ConcreteValueType::UInteger16),
            AbstractValueType::SizedUInteger(w) if *w <= 32 => Ok(ConcreteValueType::UInteger32),
            AbstractValueType::SizedUInteger(w) if *w <= 64 => Ok(ConcreteValueType::UInteger64),
            AbstractValueType::SizedUInteger(_) => Err(ReificationTooWide(*self)),
            AbstractValueType::Float => Ok(ConcreteValueType::Float32),
            AbstractValueType::SizedFloat(w) if *w <= 32 => Ok(ConcreteValueType::Float32),
            AbstractValueType::SizedFloat(w) if *w <= 64 => Ok(ConcreteValueType::Float64),
            AbstractValueType::SizedFloat(_) => Err(ReificationTooWide(*self)),
            AbstractValueType::Numeric => Err(CannotReify(*self)),
            AbstractValueType::SignedNumeric => Err(CannotReify(*self)),
            AbstractValueType::Integer => Ok(ConcreteValueType::Integer64),
            AbstractValueType::Bool => Ok(ConcreteValueType::Bool),
            AbstractValueType::Tuple(_) => Ok(ConcreteValueType::Tuple(children.to_vec())),
            AbstractValueType::Sequence => Err(CannotReify(*self)),
            AbstractValueType::String => Ok(ConcreteValueType::TString),
            AbstractValueType::Bytes => Ok(ConcreteValueType::Byte),
            AbstractValueType::Option => Ok(ConcreteValueType::Option(Box::new(children[0].clone()))),
        }
    }
}

impl ConcreteValueType {
    /// Generates a concrete type value for an annotated type.
    pub(crate) fn from_annotated_type(at: &AnnotatedType) -> Result<Self, ValueErrorKind> {
        use ValueErrorKind::*;
        match at {
            AnnotatedType::String => Ok(ConcreteValueType::TString),
            AnnotatedType::Bool => Ok(ConcreteValueType::Bool),
            AnnotatedType::Bytes => Ok(ConcreteValueType::Byte),
            AnnotatedType::Float(w) if *w <= 32 => Ok(ConcreteValueType::Float32),
            AnnotatedType::Float(w) if *w <= 64 => Ok(ConcreteValueType::Float64),
            AnnotatedType::Float(_) => Err(AnnotationTooWide(at.clone())),
            AnnotatedType::Int(w) if *w <= 8 => Ok(ConcreteValueType::Integer8),
            AnnotatedType::Int(w) if *w <= 16 => Ok(ConcreteValueType::Integer16),
            AnnotatedType::Int(w) if *w <= 32 => Ok(ConcreteValueType::Integer32),
            AnnotatedType::Int(w) if *w <= 64 => Ok(ConcreteValueType::Integer64),
            AnnotatedType::Int(_) => Err(AnnotationTooWide(at.clone())),
            AnnotatedType::UInt(w) if *w <= 8 => Ok(ConcreteValueType::UInteger8),
            AnnotatedType::UInt(w) if *w <= 16 => Ok(ConcreteValueType::UInteger16),
            AnnotatedType::UInt(w) if *w <= 32 => Ok(ConcreteValueType::UInteger32),
            AnnotatedType::UInt(w) if *w <= 64 => Ok(ConcreteValueType::UInteger64),
            AnnotatedType::UInt(_) => Err(AnnotationTooWide(at.clone())),
            AnnotatedType::Tuple(children) => {
                children
                    .iter()
                    .map(ConcreteValueType::from_annotated_type)
                    .collect::<Result<Vec<ConcreteValueType>, ValueErrorKind>>()
                    .map(ConcreteValueType::Tuple)
            },
            AnnotatedType::Option(child) => {
                ConcreteValueType::from_annotated_type(child).map(|child| ConcreteValueType::Option(Box::new(child)))
            },
            AnnotatedType::Numeric => Err(AnnotationInvalid(at.clone())),
            AnnotatedType::Sequence => Err(AnnotationInvalid(at.clone())),
            AnnotatedType::Signed => Err(AnnotationInvalid(at.clone())),
            AnnotatedType::Any => Err(AnnotationInvalid(at.clone())),
            AnnotatedType::Param(..) => Err(AnnotationInvalid(at.clone())),
        }
    }

    /// Return the width of numeric types
    pub(crate) fn width(&self) -> Option<usize> {
        use ConcreteValueType::*;
        match self {
            Bool => None,
            Integer8 => Some(8),
            Integer16 => Some(16),
            Integer32 => Some(32),
            Integer64 => Some(64),
            UInteger8 => Some(8),
            UInteger16 => Some(16),
            UInteger32 => Some(32),
            UInteger64 => Some(64),
            Float32 => Some(32),
            Float64 => Some(64),
            Tuple(_) => None,
            TString => None,
            Byte => None,
            Option(_) => None,
        }
    }
}

impl Display for AbstractValueType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractValueType::Any => write!(f, "Any"),
            AbstractValueType::Numeric => write!(f, "Numeric"),
            AbstractValueType::SignedNumeric => write!(f, "SignedNumeric"),
            AbstractValueType::Integer => write!(f, "Integer"),
            AbstractValueType::SInteger => write!(f, "Int"),
            AbstractValueType::SizedSInteger(w) => write!(f, "Int({})", *w),
            AbstractValueType::UInteger => write!(f, "UInt"),
            AbstractValueType::SizedUInteger(w) => write!(f, "UInt({})", *w),
            AbstractValueType::Float => write!(f, "Float"),
            AbstractValueType::SizedFloat(w) => write!(f, "Float({})", *w),
            AbstractValueType::Bool => write!(f, "Bool"),
            AbstractValueType::AnyTuple => write!(f, "AnyTuple"),
            AbstractValueType::Tuple(w) => write!(f, "{}Tuple", *w),
            AbstractValueType::String => write!(f, "String"),
            AbstractValueType::Bytes => write!(f, "Bytes"),
            AbstractValueType::Option => write!(f, "Option<?>"),
            AbstractValueType::Sequence => write!(f, "Sequence"),
        }
    }
}

impl Display for ConcreteValueType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConcreteValueType::Bool => write!(f, "Bool"),
            ConcreteValueType::Integer8 => write!(f, "Int8"),
            ConcreteValueType::Integer16 => write!(f, "Int16"),
            ConcreteValueType::Integer32 => write!(f, "Int32"),
            ConcreteValueType::Integer64 => write!(f, "Int64"),
            ConcreteValueType::UInteger8 => write!(f, "UInt8"),
            ConcreteValueType::UInteger16 => write!(f, "UInt16"),
            ConcreteValueType::UInteger32 => write!(f, "UInt32"),
            ConcreteValueType::UInteger64 => write!(f, "UInt64"),
            ConcreteValueType::Float32 => write!(f, "Float32"),
            ConcreteValueType::Float64 => write!(f, "Float64"),
            ConcreteValueType::Tuple(children) => {
                write!(f, "({})", children.iter().map(|c| c.to_string()).join(", "))
            },
            ConcreteValueType::TString => write!(f, "String"),
            ConcreteValueType::Byte => write!(f, "Byte"),
            ConcreteValueType::Option(c) => write!(f, "Option<{c}>"),
        }
    }
}

impl From<TcErr<AbstractValueType>> for TypeError<ValueErrorKind> {
    fn from(err: TcErr<AbstractValueType>) -> Self {
        match err {
            TcErr::KeyEquation(key1, key2, err) => {
                TypeError {
                    kind: err,
                    key1: Some(key1),
                    key2: Some(key2),
                }
            },
            TcErr::Bound(key1, key2, err) => {
                TypeError {
                    kind: err,
                    key1: Some(key1),
                    key2,
                }
            },
            TcErr::ChildAccessOutOfBound(key, value_ty, index) => {
                TypeError {
                    kind: ValueErrorKind::AccessOutOfBound(value_ty, index),
                    key1: Some(key),
                    key2: None,
                }
            },
            TcErr::ArityMismatch {
                key,
                variant,
                inferred_arity,
                reported_arity,
            } => {
                TypeError {
                    kind: ValueErrorKind::ArityMismatch(variant, inferred_arity, reported_arity),
                    key1: Some(key),
                    key2: None,
                }
            },
            TcErr::Construction(key, _, err) => {
                TypeError {
                    kind: err,
                    key1: Some(key),
                    key2: None,
                }
            },
            TcErr::ChildConstruction(key, idx, parent, err) => {
                TypeError {
                    kind: ValueErrorKind::ChildConstruction(Box::new(err), parent.variant, idx),
                    key1: Some(key),
                    key2: None,
                }
            },
            TcErr::CyclicGraph => {
                panic!("Cyclic value type constraint system");
            },
        }
    }
}

impl Resolvable for ValueErrorKind {
    fn into_diagnostic(
        self,
        spans: &[&HashMap<TcKey, Span>],
        _names: &HashMap<StreamReference, String>,
        key1: Option<TcKey>,
        key2: Option<TcKey>,
    ) -> Diagnostic {
        let spans = spans[0];
        match self {
            ValueErrorKind::TypeClash(ty1, ty2) => {
                let span1 = key1.and_then(|k| spans.get(&k).cloned());
                let span2 = key2.and_then(|k| spans.get(&k).cloned());
                Diagnostic::error(
                    &format!("In value type analysis:\nFound incompatible types: {ty1} and {ty2}"),
                )
                .maybe_add_span_with_label(span1, Some(&format!("found {ty1} here")), true)
                .maybe_add_span_with_label(span2, Some(&format!("found {ty2} here")), false)
            },
            ValueErrorKind::TupleSize(size1, size2) => {
                let span1 = key1.and_then(|k| spans.get(&k).cloned());
                let span2 = key2.and_then(|k| spans.get(&k).cloned());
                Diagnostic::error(
                    &format!(
                        "In value type analysis:\nTried to merge Tuples of different sizes {size1} and {size2}",
                    ),
                )
                .maybe_add_span_with_label(span1, Some(&format!("found Tuple of size {size1} here")), true)
                .maybe_add_span_with_label(span2, Some(&format!("found Tuple of size {size2} here")), false)
            },
            ValueErrorKind::ReificationTooWide(ty) => {
                Diagnostic::error(
                    &format!("In value type analysis:\nType {ty} is too wide to be concretized"),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
            },
            ValueErrorKind::CannotReify(ty) => {
                Diagnostic::error(
                    &format!("In value type analysis:\nType {ty} cannot be concretized"),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .add_note("Help: Consider an explicit type annotation.")
            },
            ValueErrorKind::AnnotationTooWide(ty) => {
                Diagnostic::error(
                    &format!("In value type analysis:\nAnnotated Type {ty} is too wide"),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
            },
            ValueErrorKind::AnnotationInvalid(ty) => {
                Diagnostic::error(
                    &format!("In value type analysis:\nUnknown annotated type: {ty}"),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .add_note("Help: Consider an explicit type annotation.")
            },
            ValueErrorKind::ExactTypeMismatch(inferred, expected) => {
                Diagnostic::error(
                    &format!(
                        "In value type analysis:\nInferred type {inferred} but expected {expected}.",
                    ),
                )
                .maybe_add_span_with_label(
                    key1.and_then(|k| spans.get(&k).cloned()),
                    Some(&format!("Found {expected} here")),
                    false,
                )
                .maybe_add_span_with_label(
                    key2.and_then(|k| spans.get(&k).cloned()),
                    Some(&format!("But inferred {inferred} here")),
                    true,
                )
            },
            ValueErrorKind::AccessOutOfBound(ty, idx) => {
                Diagnostic::error(
                    &format!(
                        "In value type analysis:\nChild at index {} does not exists in type {}",
                        idx - 1,
                        ty
                    ),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
            },
            ValueErrorKind::ArityMismatch(ty, inferred, expected) => {
                Diagnostic::error(
                    &format!(
                        "In value type analysis:\nExpected type {ty} to have {expected} children but inferred {inferred}",
                    ),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
            },
            ValueErrorKind::ChildConstruction(child_err, parent, idx) => {
                let reason = match child_err.as_ref() {
                    ValueErrorKind::ReificationTooWide(ty) => format!("Type {ty} is too wide to be concretized"),
                    ValueErrorKind::CannotReify(ty) => format!("Type {ty} cannot be concretized"),
                    _ => "Unknown".to_string(),
                };
                Diagnostic::error(
                    &format!("In value type analysis:\nCannot construct sub type of {parent} at index {idx}.\nReason: {reason}"),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
            },
            ValueErrorKind::UnnecessaryTypeParam(span) => {
                Diagnostic::error(
                    "This function has more input type parameter then defined generic types. All unnecessary type arguments can be removed.",
                )
                    .add_span_with_label(span, Some("here"), true)
            },
            ValueErrorKind::InvalidWiden(bound, inner) => {
                Diagnostic::error(
                    &format!("In value type analysis:\nInvalid application of the widen operator.\nTarget width is {} but supplied width is {}.", bound.width().unwrap_or(0), inner.width().unwrap_or(0))
                )
                    .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some(&format!("Widen with traget {bound} is found here")), false)
                    .maybe_add_span_with_label(key2.and_then(|k| spans.get(&k).cloned()), Some(&format!("Inferred type {inner} here")), true)
            },
            ValueErrorKind::OptionNotAllowed(ty) => {
                Diagnostic::error(
                "In value type analysis:\nAn optional type is not allowed here."
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some(&format!("Optional type: {ty} found here")), true)
                    .add_note("Help: Consider using the default operator to resolve the optional.")
            }
            ValueErrorKind::WrongTriggerMsg(ty) => {
                Diagnostic::error("In value type analysis:\nAn trigger message has to be of type string.").maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some(&format!("Found {ty} here.")), true)
            }
        }
    }
}
