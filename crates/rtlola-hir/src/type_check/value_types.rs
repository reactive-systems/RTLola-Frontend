use std::cmp::max;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use itertools::Itertools;
use rtlola_reporting::{Diagnostic, Span};
use rusttyc::{Arity, Constructable, Partial, TcErr, TcKey, Variant};

use super::*;
use crate::hir::{AnnotatedType, StreamReference};
use crate::type_check::pacing_types::Freq;
use crate::type_check::rtltc::{Emittable, TypeError};

#[derive(Debug, Clone)]
pub(crate) enum ValueErrorKind {
    TypeClash(AbstractValueType, AbstractValueType),
    TupleSize(usize, usize),
    ReificationTooWide(AbstractValueType),
    CannotReify(AbstractValueType),
    AnnotationTooWide(AnnotatedType),
    AnnotationInvalid(AnnotatedType),
    ///target freq, Offset
    IncompatibleRealTimeOffset(Freq, i64),
    /// Inferred, Expected
    ExactTypeMismatch(ConcreteValueType, ConcreteValueType),
    AccessOutOfBound(AbstractValueType, usize),
    /// Type, inferred, reported
    ArityMismatch(AbstractValueType, usize, usize),
    /// Child Construction Error, Parent Type, Index of Child
    ChildConstruction(Box<Self>, AbstractValueType, usize),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum AbstractValueType {
    Any,
    Numeric,
    Integer,
    SInteger(u32),
    UInteger(u32),
    Float(u32),
    Bool,
    AnyTuple,
    Tuple(usize),
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
            (Integer, Integer) => Ok((Integer, 0)),
            (SInteger(l), SInteger(r)) => Ok((SInteger(max(r, l)), 0)),
            (UInteger(l), UInteger(r)) => Ok((UInteger(max(r, l)), 0)),
            (Float(l), Float(r)) => Ok((Float(max(l, r)), 0)),
            (Bool, Bool) => Ok((Bool, 0)),
            (Bool, _) | (_, Bool) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Numeric, Integer) | (Integer, Numeric) => Ok((Integer, 0)),
            (Numeric, SInteger(w)) | (SInteger(w), Numeric) => Ok((SInteger(w), 0)),
            (Numeric, UInteger(w)) | (UInteger(w), Numeric) => Ok((UInteger(w), 0)),
            (Numeric, Float(i)) | (Float(i), Numeric) => Ok((Float(i), 0)),
            (Integer, SInteger(x)) | (SInteger(x), Integer) => Ok((SInteger(x), 0)),
            (Integer, UInteger(x)) | (UInteger(x), Integer) => Ok((UInteger(x), 0)),
            (Integer, _) | (_, Integer) => Err(TypeClash(lhs.variant, rhs.variant)),
            (SInteger(_), _) | (_, SInteger(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (UInteger(_), _) | (_, UInteger(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
            (AnyTuple, AnyTuple) => Ok((AnyTuple, max(lhs.least_arity, rhs.least_arity))),
            (AnyTuple, Tuple(size)) => tuple_meet(lhs.least_arity, size),
            (Tuple(size), AnyTuple) => tuple_meet(rhs.least_arity, size),
            (AnyTuple, _) | (_, AnyTuple) => Err(TypeClash(lhs.variant, rhs.variant)),
            (Tuple(size_l), Tuple(size_r)) if size_l == size_r => Ok((Tuple(size_l), size_l)),
            (Tuple(size_l), Tuple(size_r)) => Err(TupleSize(size_l, size_r)),
            (Tuple(_), _) | (_, Tuple(_)) => Err(TypeClash(lhs.variant, rhs.variant)),
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
            Numeric | Integer | SInteger(_) | UInteger(_) | Float(_) | Bool | String | Bytes => Arity::Fixed(0),
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
            AbstractValueType::SInteger(w) if *w <= 8 => Ok(ConcreteValueType::Integer8),
            AbstractValueType::SInteger(w) if *w <= 16 => Ok(ConcreteValueType::Integer16),
            AbstractValueType::SInteger(w) if *w <= 32 => Ok(ConcreteValueType::Integer32),
            AbstractValueType::SInteger(w) if *w <= 64 => Ok(ConcreteValueType::Integer64),
            AbstractValueType::SInteger(_) => Err(ReificationTooWide(*self)),
            AbstractValueType::UInteger(w) if *w <= 8 => Ok(ConcreteValueType::UInteger8),
            AbstractValueType::UInteger(w) if *w <= 16 => Ok(ConcreteValueType::UInteger16),
            AbstractValueType::UInteger(w) if *w <= 32 => Ok(ConcreteValueType::UInteger32),
            AbstractValueType::UInteger(w) if *w <= 64 => Ok(ConcreteValueType::UInteger64),
            AbstractValueType::UInteger(_) => Err(ReificationTooWide(*self)),
            AbstractValueType::Float(w) if *w <= 32 => Ok(ConcreteValueType::Float32),
            AbstractValueType::Float(w) if *w <= 64 => Ok(ConcreteValueType::Float64),
            AbstractValueType::Float(_) => Err(ReificationTooWide(*self)),
            AbstractValueType::Numeric => Err(CannotReify(*self)),
            AbstractValueType::Integer => Ok(ConcreteValueType::Integer64),
            AbstractValueType::Bool => Ok(ConcreteValueType::Bool),
            AbstractValueType::Tuple(_) => Ok(ConcreteValueType::Tuple(children.to_vec())),
            AbstractValueType::String => Ok(ConcreteValueType::TString),
            AbstractValueType::Bytes => Ok(ConcreteValueType::Byte),
            AbstractValueType::Option => Ok(ConcreteValueType::Option(Box::new(children[0].clone()))),
        }
    }
}

impl ConcreteValueType {
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

            AnnotatedType::Param(..) => Err(AnnotationInvalid(at.clone())),
        }
    }
}

impl Display for AbstractValueType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractValueType::Any => write!(f, "Any"),
            AbstractValueType::Numeric => write!(f, "Numeric"),
            AbstractValueType::Integer => write!(f, "Integer"),
            AbstractValueType::SInteger(w) => write!(f, "Int({})", *w),
            AbstractValueType::UInteger(w) => write!(f, "UInt({})", *w),
            AbstractValueType::Float(w) => write!(f, "Float({})", *w),
            AbstractValueType::Bool => write!(f, "Bool"),
            AbstractValueType::AnyTuple => write!(f, "AnyTuple"),
            AbstractValueType::Tuple(w) => write!(f, "{}Tuple", *w),
            AbstractValueType::String => write!(f, "String"),
            AbstractValueType::Bytes => write!(f, "String"),
            AbstractValueType::Option => write!(f, "Option<?>"),
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
            ConcreteValueType::Option(c) => write!(f, "Option<{}>", c),
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
        }
    }
}

impl Emittable for ValueErrorKind {
    fn emit(
        self,
        handler: &Handler,
        spans: &[&HashMap<TcKey, Span>],
        _names: &HashMap<StreamReference, &str>,
        key1: Option<TcKey>,
        key2: Option<TcKey>,
    ) {
        let spans = spans[0];
        match self {
            ValueErrorKind::TypeClash(ty1, ty2) => {
                let span1 = key1.and_then(|k| spans.get(&k).cloned());
                let span2 = key2.and_then(|k| spans.get(&k).cloned());
                Diagnostic::error(
                    handler,
                    &format!("In value type analysis:\nFound incompatible types: {} and {}", ty1, ty2),
                )
                .maybe_add_span_with_label(span1, Some(&format!("found {} here", ty1)), true)
                .maybe_add_span_with_label(span2, Some(&format!("found {} here", ty2)), false)
                .emit()
            },
            ValueErrorKind::TupleSize(size1, size2) => {
                let span1 = key1.and_then(|k| spans.get(&k).cloned());
                let span2 = key2.and_then(|k| spans.get(&k).cloned());
                Diagnostic::error(
                    handler,
                    &format!(
                        "In value type analysis:\nTried to merge Tuples of different sizes {} and {}",
                        size1, size2
                    ),
                )
                .maybe_add_span_with_label(span1, Some(&format!("found Tuple of size {} here", size1)), true)
                .maybe_add_span_with_label(span2, Some(&format!("found Tuple of size {} here", size2)), false)
                .emit()
            },
            ValueErrorKind::ReificationTooWide(ty) => {
                Diagnostic::error(
                    handler,
                    &format!("In value type analysis:\nType {} is too wide to be concretized", ty),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .emit()
            },
            ValueErrorKind::CannotReify(ty) => {
                Diagnostic::error(
                    handler,
                    &format!("In value type analysis:\nType {} cannot be concretized", ty),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .add_note("Help: Consider an explicit type annotation.")
                .emit()
            },
            ValueErrorKind::AnnotationTooWide(ty) => {
                Diagnostic::error(
                    handler,
                    &format!("In value type analysis:\nAnnotated Type {} is too wide", ty),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .emit()
            },
            ValueErrorKind::AnnotationInvalid(ty) => {
                Diagnostic::error(
                    handler,
                    &format!("In value type analysis:\nUnknown annotated type: {}", ty),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .add_note("Help: Consider an explicit type annotation.")
                .emit()
            },
            ValueErrorKind::IncompatibleRealTimeOffset(freq, dur) => {
                Diagnostic::error(
                    handler,
                    "In value type analysis:\nReal-Time offset is incompatible with the frequency of the target stream",
                )
                .maybe_add_span_with_label(
                    key1.and_then(|k| spans.get(&k).cloned()),
                    Some(&format!("Found offset with duration {} here", dur)),
                    true,
                )
                .maybe_add_span_with_label(
                    key2.and_then(|k| spans.get(&k).cloned()),
                    Some(&format!("Target stream with frequency {} is found here", freq)),
                    false,
                )
                .emit()
            },

            ValueErrorKind::ExactTypeMismatch(inferred, expected) => {
                Diagnostic::error(
                    handler,
                    &format!(
                        "In value type analysis:\nInferred type {} but expected {}.",
                        inferred, expected
                    ),
                )
                .maybe_add_span_with_label(
                    key1.and_then(|k| spans.get(&k).cloned()),
                    Some(&format!("Expected {} here", expected)),
                    true,
                )
                .maybe_add_span_with_label(
                    key2.and_then(|k| spans.get(&k).cloned()),
                    Some(&format!("But inferred {} here", inferred)),
                    false,
                )
                .emit()
            },
            ValueErrorKind::AccessOutOfBound(ty, idx) => {
                Diagnostic::error(
                    handler,
                    &format!(
                        "In value type analysis:\nChild at index {} does not exists in type {}",
                        idx - 1,
                        ty
                    ),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .emit()
            },
            ValueErrorKind::ArityMismatch(ty, inferred, expected) => {
                Diagnostic::error(
                    handler,
                    &format!(
                        "In value type analysis:\nExpected type {} to have {} children but inferred {}",
                        ty, expected, inferred
                    ),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .emit()
            },
            ValueErrorKind::ChildConstruction(child_err, parent, idx) => {
                let reason = match child_err.as_ref() {
                    ValueErrorKind::ReificationTooWide(ty) => format!("Type {} is too wide to be concretized", ty),
                    ValueErrorKind::CannotReify(ty) => format!("Type {} cannot be concretized", ty),
                    _ => "Unknown".to_string(),
                };
                Diagnostic::error(
                    handler,
                    &format!(
                        "In value type analysis:\nCannot construct sub type of {} at index {}.\nReason: {}",
                        parent, idx, reason
                    ),
                )
                .maybe_add_span_with_label(key1.and_then(|k| spans.get(&k).cloned()), Some("here"), true)
                .emit()
            },
        }
    }
}
