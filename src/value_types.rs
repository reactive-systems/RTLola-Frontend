use super::*;
use std::cmp::max;
use rusttyc::types::{ReificationErr,Abstract};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IAbstractType {
    Any,
    Numeric,
    Integer(u32),
    UInteger(u32),
    Float(u32),
    Bool,
    Tuple(Vec<IAbstractType>),
    TString,
    Option(Box<IAbstractType>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RecursiveType {
    Option,
    Tuple(u8),
    Other,
}

impl Abstract for IAbstractType {
    type Err = String;

    fn unconstrained() -> Self {
        IAbstractType::Any
    }

    fn meet(&self, other: &Self) -> Result<Self, Self::Err> {
        use IAbstractType::*;
        match (self, other) {
            (Any, other) | (other, Any) => Ok(other.clone()),
            (Integer(l), Integer(r)) => Ok(Integer(max(*r, *l))),
            (UInteger(l), UInteger(r)) => Ok(UInteger(max(*r, *l))),
            (Float(l), Float(r)) => Ok(Float(max(*l, *r))),
            (Float(i), Integer(u)) | (Integer(u), Float(i)) => Ok(Float(max(*i, *u))),
            (Bool, Bool) => Ok(Bool),
            (Bool, other) | (other, Bool) => {
                Err(String::from(format!("Bool not unifiable with {:?}", other)))
            }
            (Numeric, Integer(w)) | (Integer(w), Numeric) => Ok(Integer(*w)),
            (Numeric, UInteger(w)) | (UInteger(w), Numeric) => Ok(UInteger(*w)),
            (Numeric, Float(i)) | (Float(i), Numeric) => Ok(Float(*i)),
            (Numeric, Numeric) => Ok(Numeric),
            (UInteger(u), other) | (other, UInteger(u)) => {
                Err(String::from(format!("UInt not unifiable with {:?}", other)))
            }
            (Tuple(lv), Tuple(rv)) => {
                if lv.len() != rv.len() {
                    return Err(String::from(
                        "Tuple unification demands equal number of arguments",
                    ));
                }
                let (recursive_result, errors): (Vec<Result<Self, Self::Err>>, Vec<_>) = lv
                    .iter()
                    .zip(rv.iter())
                    .map(|(l, r)| Self::meet(l, r))
                    .partition(Result::is_ok);
                if errors.len() > 0 {
                    let error_unwraped: Vec<String> =
                        errors.into_iter().map(Result::unwrap_err).collect();
                    return Err(error_unwraped.join(", "));
                } else {
                    Ok(Tuple(
                        recursive_result.into_iter().map(Result::unwrap).collect(),
                    ))
                }
            }
            (Tuple(_), _) | (_, Tuple(_)) => {
                Err(String::from("Tuple unification only with other Tuples"))
            }
            (TString, TString) => Ok(TString),
            (TString, _) | (_, TString) => {
                Err(String::from("String unification only with other Strings"))
            }
            (Option(l), Option(r)) => match Self::meet(l, r) {
                Ok(t) => Ok(Option(Box::new(t))),
                Err(e) => Err(e),
            },
            (Option(_), _) | (_, Option(_)) => {
                Err(String::from("Option unification only with other Options"))
            } //(l, r) => Err(String::from(format!("unification error: left: {:?}, right: {:?}",l,r))),
        }
    }

    fn arity(&self) -> std::option::Option<usize> {
        use IAbstractType::*;
        match self {
            Option(_) => Some(1),
            Tuple(t) => Some(t.len()),
            Any |
            Numeric |
            Integer(_) |
            UInteger(_) |
            Float(_) |
            Bool |
            TString => Some(0),
        }
    }

    fn nth_child(&self, n: usize) -> &Self {
        use IAbstractType::*;
        match self {
            Option(op) => &*op,
            Tuple(vec) => &vec[n],
            Any | Numeric | Integer(_) | UInteger(_) | Float(_) | Bool | TString => unreachable!()
        }
    }

    fn with_children<I>(&self, children: I) -> Self where
        I: IntoIterator<Item=Self> {
        let mut it = children.into_iter();
        match self {
            IAbstractType::Option(op ) => IAbstractType::Option(it.next().expect("").into()),
            IAbstractType::Tuple(v) => IAbstractType::Tuple(it.collect()),
            t => t.clone(),
        }
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
    Option(Box<IConcreteType>),
}
use IConcreteType::*;

impl rusttyc::types::TryReifiable for IAbstractType {
    type Reified = IConcreteType;

    fn try_reify(&self) -> Result<Self::Reified, ReificationErr> {
        match self {
            IAbstractType::Any => Err(ReificationErr::TooGeneral("Cannot reify `Any`.".to_string())),
            IAbstractType::Integer(w) if *w <= 8 => Ok(IConcreteType::Integer8),
            IAbstractType::Integer(w) if *w <= 16 => Ok(IConcreteType::Integer16),
            IAbstractType::Integer(w) if *w <= 32 => Ok(IConcreteType::Integer32),
            IAbstractType::Integer(w) if *w <= 64 => Ok(IConcreteType::Integer64),
            IAbstractType::Integer(w) => {
                Err(ReificationErr::Conflicting(format!("Integer too wide, {}-bit not supported.", w)))
            }
            IAbstractType::UInteger(w) if *w <= 8  => Ok(IConcreteType::UInteger8),
            IAbstractType::UInteger(w) if *w <= 16 => Ok(IConcreteType::UInteger16),
            IAbstractType::UInteger(w) if *w <= 32 => Ok(IConcreteType::UInteger32),
            IAbstractType::UInteger(w) if *w <= 64 => Ok(IConcreteType::UInteger64),
            IAbstractType::UInteger(w) => {
                Err(ReificationErr::Conflicting(format!("UInteger too wide, {}-bit not supported.", w)))
            }
            IAbstractType::Float(w) if *w <= 32 => Ok(IConcreteType::Float32),
            IAbstractType::Float(w) if *w <= 64 => Ok(IConcreteType::Float64),
            IAbstractType::Float(w) => {
                Err(ReificationErr::Conflicting(format!("Floating point number too wide, {}-bit not supported.", w)))
            }
            IAbstractType::Numeric => Err(ReificationErr::TooGeneral(
                "Cannot reify a numeric value. Either define a default (int/fixed) or restrict type.".to_string(),
            )),
            IAbstractType::Bool => Ok(IConcreteType::Bool),
            IAbstractType::Tuple(sub_types) => {
                let (recursive_result, errors): (Vec<Result<IConcreteType, ReificationErr>>, Vec<_>) = sub_types
                    .iter()
                    .map(|v| rusttyc::types::TryReifiable::try_reify(v))
                    .partition(Result::is_ok);
                if errors.len() > 0 {
                    let error_unwraped: Vec<ReificationErr> =
                        errors.into_iter().map(Result::unwrap_err).collect();
                    return Err(match &error_unwraped[0] {
                        ReificationErr::Conflicting(s) => ReificationErr::Conflicting(s.clone()),
                        ReificationErr::TooGeneral(s) => ReificationErr::TooGeneral(s.clone()),
                    })
                } else {
                    Ok(Tuple(
                        recursive_result.into_iter().map(Result::unwrap).collect(),
                    ))
                }
            }
            IAbstractType::TString => Ok(IConcreteType::TString),
            IAbstractType::Option(inner_type) => {
                let res: Result<IConcreteType, ReificationErr> = Self::try_reify(&**inner_type);
                match res {
                    Err(e) => Err(e),
                    Ok(ty) => Ok(IConcreteType::Option(Box::new(ty))),
                }
            }
        }
    }
}

impl rusttyc::types::Generalizable for IConcreteType {
    type Generalized = IAbstractType;

    fn generalize(&self) -> Self::Generalized {
        match self {
            IConcreteType::Float64 => IAbstractType::Float(64),
            IConcreteType::Float32 => IAbstractType::Float(32),
            IConcreteType::Integer8 => IAbstractType::Integer(8),
            IConcreteType::Integer16 => IAbstractType::Integer(16),
            IConcreteType::Integer32 => IAbstractType::Integer(32),
            IConcreteType::Integer64 => IAbstractType::Integer(64),
            IConcreteType::UInteger8 => IAbstractType::UInteger(8),
            IConcreteType::UInteger16 => IAbstractType::UInteger(16),
            IConcreteType::UInteger32 => IAbstractType::UInteger(32),
            IConcreteType::UInteger64 => IAbstractType::UInteger(64),
            IConcreteType::Bool => IAbstractType::Bool,
            IConcreteType::Tuple(type_list) => {
                let result_vec: Vec<IAbstractType> =
                    type_list.into_iter().map(|t| Self::generalize(t)).collect();
                IAbstractType::Tuple(result_vec)
            }
            IConcreteType::TString => IAbstractType::TString,
            IConcreteType::Option(inner_type) => {
                let result: IAbstractType = Self::generalize(&**inner_type);
                IAbstractType::Option(Box::new(result))
            }
        }
    }
}
