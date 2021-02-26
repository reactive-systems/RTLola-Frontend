use super::rusttyc::{Arity, Partial};
use super::*;
use rusttyc::{Constructable, Variant};
use std::cmp::max;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum IAbstractType {
    Any,
    Numeric,
    Integer,
    SInteger(u32),
    UInteger(u32),
    Float(u32),
    Bool,
    AnyTuple,
    Tuple(usize),
    TString,
    Bytes,
    Option,
}

impl Variant for IAbstractType {
    type Err = String;

    fn meet(lhs: Partial<IAbstractType>, rhs: Partial<IAbstractType>) -> Result<Partial<IAbstractType>, Self::Err> {
        fn tuple_meet(least_arity: usize, size: usize) -> Result<(IAbstractType, usize), String> {
            if least_arity <= size {
                Ok((Tuple(size), size))
            } else {
                Err(format!("AnyTuple of length {} cannot be unified with tuple of length {}", least_arity, size))
            }
        }
        use IAbstractType::*;
        dbg!(&rhs, &lhs);
        let (new_var, min_arity) = match (lhs.variant, rhs.variant) {
            (Any, other) => Ok((other.clone(), rhs.least_arity)),
            (other, Any) => Ok((other.clone(), lhs.least_arity)),
            (Numeric, Numeric) => Ok((Numeric, 0)),
            (Integer, Integer) => Ok((Integer, 0)),
            (SInteger(l), SInteger(r)) => Ok((SInteger(max(r, l)), 0)),
            (UInteger(l), UInteger(r)) => Ok((UInteger(max(r, l)), 0)),
            (Float(l), Float(r)) => Ok((Float(max(l, r)), 0)),
            (Bool, Bool) => Ok((Bool, 0)),
            (Bool, other) | (other, Bool) => Err(format!("Bool not unifiable with {:?}", other)),
            (Numeric, Integer) | (Integer, Numeric) => Ok((Integer, 0)),
            (Numeric, SInteger(w)) | (SInteger(w), Numeric) => Ok((SInteger(w), 0)),
            (Numeric, UInteger(w)) | (UInteger(w), Numeric) => Ok((UInteger(w), 0)),
            (Numeric, Float(i)) | (Float(i), Numeric) => Ok((Float(i), 0)),
            (Integer, SInteger(x)) | (SInteger(x), Integer) => Ok((SInteger(x), 0)),
            (Integer, UInteger(x)) | (UInteger(x), Integer) => Ok((UInteger(x), 0)),
            (Integer, other) | (other, Integer) => Err(format!("Integer and non-Integer {:?}", other)),
            (SInteger(_), other) | (other, SInteger(_)) => Err(format!("Int not unifiable with {:?}", other)),
            (UInteger(_), other) | (other, UInteger(_)) => Err(format!("UInt not unifiable with {:?}", other)),
            (AnyTuple, AnyTuple) => Ok((AnyTuple, max(lhs.least_arity, rhs.least_arity))),
            (AnyTuple, Tuple(size)) => tuple_meet(lhs.least_arity, size),
            (Tuple(size), AnyTuple) => tuple_meet(rhs.least_arity, size),
            (AnyTuple, _) | (_, AnyTuple) => Err(String::from("Tuple unification only with other Tuples")),
            (Tuple(size_l), Tuple(size_r)) => {
                if size_l == size_r {
                    Ok((Tuple(size_l), size_l))
                } else {
                    Err(format!("Tuple of length {} cannot be unified with tuple of length {}", size_l, size_r))
                }
            }
            (Tuple(_), _) | (_, Tuple(_)) => Err(String::from("Tuple unification only with other Tuples")),
            (TString, TString) => Ok((TString, 0)),
            (TString, _) | (_, TString) => Err(String::from("String unification only with other Strings")),
            (Bytes, Bytes) => Ok((Bytes, 0)),
            (Bytes, _) | (_, Bytes) => Err(String::from("Bytes unification only with other Bytes")),
            (Option, Option) => Ok((Option, 1)),
            (Option, _) | (_, Option) => Err(String::from("Option unification only with other Options")), //(l, r) => Err(String::from(format!("unification error: left: {:?}, right: {:?}",l,r))),
        }?;
        Ok(Partial { variant: new_var, least_arity: min_arity })
    }

    fn arity(&self) -> Arity {
        use IAbstractType::*;
        match self {
            Any | AnyTuple => Arity::Variable,
            Tuple(x) => Arity::Fixed(*x),
            Option => Arity::Fixed(1),
            _ => Arity::Fixed(0),
        }
    }

    fn top() -> Self {
        Self::Any
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IConcreteType {
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
    Tuple(Vec<IConcreteType>),
    TString,
    Byte,
    Option(Box<IConcreteType>),
}

impl Constructable for IAbstractType {
    type Type = IConcreteType;

    fn construct(&self, children: &[IConcreteType]) -> Result<IConcreteType, String> {
        match self {
            IAbstractType::Any => Err("Cannot reify `Any`.".to_string()),
            IAbstractType::AnyTuple => Err("Cannot reify AnyTuple".to_string()),
            IAbstractType::SInteger(w) if *w <= 8 => Ok(IConcreteType::Integer8),
            IAbstractType::SInteger(w) if *w <= 16 => Ok(IConcreteType::Integer16),
            IAbstractType::SInteger(w) if *w <= 32 => Ok(IConcreteType::Integer32),
            IAbstractType::SInteger(w) if *w <= 64 => Ok(IConcreteType::Integer64),
            IAbstractType::SInteger(w) => Err(format!("Integer too wide, {}-bit not supported.", w)),
            IAbstractType::UInteger(w) if *w <= 8 => Ok(IConcreteType::UInteger8),
            IAbstractType::UInteger(w) if *w <= 16 => Ok(IConcreteType::UInteger16),
            IAbstractType::UInteger(w) if *w <= 32 => Ok(IConcreteType::UInteger32),
            IAbstractType::UInteger(w) if *w <= 64 => Ok(IConcreteType::UInteger64),
            IAbstractType::UInteger(w) => Err(format!("UInteger too wide, {}-bit not supported.", w)),
            IAbstractType::Float(w) if *w <= 32 => Ok(IConcreteType::Float32),
            IAbstractType::Float(w) if *w <= 64 => Ok(IConcreteType::Float64),
            IAbstractType::Float(w) => Err(format!("Floating point number too wide, {}-bit not supported.", w)),
            IAbstractType::Numeric => {
                Err("Cannot reify a numeric value. Either define a default (int/fixed) or restrict type.".to_string())
            }
            /*
            IAbstractType::Integer => Err(ReificationErr::TooGeneral(
                "Cannot reify an Integer value. Either define a default (int/uint) or restrict type.".to_string(),
            )),
            */
            IAbstractType::Integer => Ok(IConcreteType::Integer32), //TODO REVIEW default case
            IAbstractType::Bool => Ok(IConcreteType::Bool),
            IAbstractType::Tuple(_) => Ok(IConcreteType::Tuple(children.to_vec())),
            IAbstractType::TString => Ok(IConcreteType::TString),
            IAbstractType::Bytes => Ok(IConcreteType::Byte),
            IAbstractType::Option => Ok(IConcreteType::Option(Box::new(children[0].clone()))),
        }
    }
}

// impl rusttyc::types::Generalizable for IConcreteType {
//     type Generalized = IAbstractType;
//
//     fn generalize(&self) -> Self::Generalized {
//         match self {
//             IConcreteType::Float64 => IAbstractType::Float(64),
//             IConcreteType::Float32 => IAbstractType::Float(32),
//             IConcreteType::Integer8 => IAbstractType::SInteger(8),
//             IConcreteType::Integer16 => IAbstractType::SInteger(16),
//             IConcreteType::Integer32 => IAbstractType::SInteger(32),
//             IConcreteType::Integer64 => IAbstractType::SInteger(64),
//             IConcreteType::UInteger8 => IAbstractType::UInteger(8),
//             IConcreteType::UInteger16 => IAbstractType::UInteger(16),
//             IConcreteType::UInteger32 => IAbstractType::UInteger(32),
//             IConcreteType::UInteger64 => IAbstractType::UInteger(64),
//             IConcreteType::Bool => IAbstractType::Bool,
//             IConcreteType::Tuple(type_list) => {
//                 let result_vec: Vec<IAbstractType> = type_list.iter().map(|t| Self::generalize(t)).collect();
//                 IAbstractType::Tuple(result_vec)
//             }
//             IConcreteType::TString => IAbstractType::TString,
//             IConcreteType::Byte => IAbstractType::Bytes,
//             IConcreteType::Option(inner_type) => {
//                 let result: IAbstractType = Self::generalize(&**inner_type);
//                 IAbstractType::Option(Box::new(result))
//             }
//         }
//     }
// }
