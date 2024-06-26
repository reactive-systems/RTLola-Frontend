//! This module contains the Lola standard library.
use std::iter;

use lazy_static::lazy_static;

use crate::hir::{AnnotatedType, FunctionName};

/// The types of the function arguments
#[derive(Debug, Clone)]
pub enum ParameterDecl {
    /// The function declaration has a fixed number of parameters with the following types
    FixedAmount(Vec<AnnotatedType>),
    /// The function declaration has an arbitary number of parameters, first with a fixed
    /// amount of parameters of a given type, then an arbitary amount of parameters of the same type
    ArbitaryAmount {
        fixed: Vec<AnnotatedType>,
        repeating: AnnotatedType,
    },
}

impl ParameterDecl {
    /// returns an (possibly infinite) iterator over the types of the parameters
    pub(crate) fn iter(&self) -> Box<dyn Iterator<Item = &AnnotatedType> + '_> {
        match self {
            ParameterDecl::FixedAmount(v) => Box::new(v.iter()),
            ParameterDecl::ArbitaryAmount { fixed, repeating } => Box::new(fixed.iter().chain(iter::repeat(repeating))),
        }
    }
}

/// A (possibly generic) function declaration
#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: FunctionName,
    pub(crate) generics: Vec<AnnotatedType>,
    pub(crate) parameters: ParameterDecl,
    pub(crate) return_type: AnnotatedType,
}

lazy_static! {
    // fn widen_signed<T: Signed, U: Signed>(T) -> U where U: T
    static ref WIDEN: FuncDecl = FuncDecl {
        name: FunctionName::new("widen".to_string(), &[None]),
        generics: vec![AnnotatedType::Numeric,AnnotatedType::Numeric],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(1, "U".to_string()),
    };
    // fn sqrt<T: FloatingPoint>(T) -> T
    static ref SQRT: FuncDecl = FuncDecl {
        name: FunctionName::new("sqrt".to_string(), &[None]),
        generics: vec![AnnotatedType::Float(0)],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn min<T: Numeric>(T, T) -> T
    static ref MIN: FuncDecl = FuncDecl {
        name: FunctionName::new("min".to_string(), &[None, None]),
        generics: vec![AnnotatedType::Numeric],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string()), AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn max<T: Numeric>(T, T) -> T
    static ref MAX: FuncDecl = FuncDecl {
        name: FunctionName::new("max".to_string(), &[None, None]),
        generics: vec![AnnotatedType::Numeric],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string()), AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn cos<T: FloatingPoint>(T) -> T
    static ref COS: FuncDecl = FuncDecl {
        name: FunctionName::new("cos".to_string(), &[None]),
        generics: vec![AnnotatedType::Float(0),
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn sin<T: FloatingPoint>(T) -> T
    static ref SIN: FuncDecl = FuncDecl {
        name: FunctionName::new("sin".to_string(), &[None]),
        generics: vec![AnnotatedType::Float(0),
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn tan<T: FloatingPoint>(T) -> T
    static ref TAN: FuncDecl = FuncDecl {
        name: FunctionName::new("tan".to_string(), &[None]),
        generics: vec![AnnotatedType::Float(0),
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn arcsin<T: FloatingPoint>(T) -> T
    static ref ARCSIN: FuncDecl = FuncDecl {
        name: FunctionName::new("arcsin".to_string(), &[None]),
        generics: vec![ AnnotatedType::Float(0),
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn arccos<T: FloatingPoint>(T) -> T
    static ref ARCCOS: FuncDecl = FuncDecl {
        name: FunctionName::new("arccos".to_string(), &[None]),
        generics: vec![ AnnotatedType::Float(0),
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn arctan<T: FloatingPoint>(T) -> T
    static ref ARCTAN: FuncDecl = FuncDecl {
        name: FunctionName::new("arctan".to_string(), &[None]),
        generics: vec![ AnnotatedType::Float(0),
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn abs<T: Numeric>(T) -> T
    static ref ABS: FuncDecl = FuncDecl {
        name: FunctionName::new("abs".to_string(), &[None]),
        generics: vec![AnnotatedType::Signed,
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };

    // fn matches<T: Sequence>(T, regex: String) -> Bool
    static ref MATCHES: FuncDecl = FuncDecl {
        name: FunctionName::new("matches".to_string(), &[None, Some("regex".to_string())]),
        generics: vec![AnnotatedType::Sequence],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string()), AnnotatedType::String]),
        return_type: AnnotatedType::Bool,
    };

    /// fn cast<T: Numeric, U: Numeric>(T) -> U
    /// allows for arbitrary conversion of numeric types T -> U
    static ref CAST: FuncDecl = FuncDecl {
        name: FunctionName::new("cast".to_string(), &[None]),
        generics: vec![AnnotatedType::Numeric,
             AnnotatedType::Numeric,
        ],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Param(0, "T".to_string())]),
        return_type: AnnotatedType::Param(1, "U".to_string()),
    };

    /// access index of byte array
    static ref BYTES_AT: FuncDecl = FuncDecl {
        name: FunctionName::new("at".to_string(), &[None, Some("index".to_string())]),
        generics: vec![],
        parameters: ParameterDecl::FixedAmount(vec![AnnotatedType::Bytes, AnnotatedType::UInt(8)]),
        return_type: AnnotatedType::Option(AnnotatedType::UInt(8).into()),
    };

    /// format a bunch of values with a format string
    static ref FORMAT: FuncDecl = FuncDecl {
        name: FunctionName::new_repeating("format".to_string()),
        generics: vec![],
        parameters: ParameterDecl::ArbitaryAmount{fixed: vec![AnnotatedType::String], repeating: AnnotatedType::Any},
        return_type: AnnotatedType::String
    };
}

pub(crate) fn implicit_module() -> Vec<&'static FuncDecl> {
    vec![&WIDEN, &CAST, &BYTES_AT, &FORMAT]
}

pub(crate) fn math_module() -> Vec<&'static FuncDecl> {
    vec![&SQRT, &COS, &SIN, &TAN, &ARCSIN, &ARCCOS, &ARCTAN, &ABS, &MIN, &MAX]
}

pub(crate) fn regex_module() -> Vec<&'static FuncDecl> {
    vec![&MATCHES]
}

lazy_static! {
    pub(crate) static ref PRIMITIVE_TYPES: Vec<(&'static str, &'static AnnotatedType)> = vec![
        ("Bool", &AnnotatedType::Bool),
        ("Int8", &AnnotatedType::Int(8)),
        ("Int16", &AnnotatedType::Int(16)),
        ("Int32", &AnnotatedType::Int(32)),
        ("Int64", &AnnotatedType::Int(64)),
        ("UInt8", &AnnotatedType::UInt(8)),
        ("UInt16", &AnnotatedType::UInt(16)),
        ("UInt32", &AnnotatedType::UInt(32)),
        ("UInt64", &AnnotatedType::UInt(64)),
        ("Float16", &AnnotatedType::Float(16)),
        ("Float32", &AnnotatedType::Float(32)),
        ("Float64", &AnnotatedType::Float(64)),
        ("String", &AnnotatedType::String),
        ("Bytes", &AnnotatedType::Bytes),
    ];
    pub(crate) static ref REDUCED_PRIMITIVE_TYPES: Vec<(&'static str, &'static AnnotatedType)> = vec![
        ("Bool", &AnnotatedType::Bool),
        ("Int64", &AnnotatedType::Int(64)),
        ("UInt64", &AnnotatedType::UInt(64)),
        ("Float64", &AnnotatedType::Float(64)),
        ("String", &AnnotatedType::String),
        ("Bytes", &AnnotatedType::Bytes),
    ];
    pub(crate) static ref PRIMITIVE_TYPES_ALIASES: Vec<(&'static str, &'static AnnotatedType)> = vec![
        ("Int", &AnnotatedType::Int(64)),
        ("UInt", &AnnotatedType::UInt(64)),
        ("Float", &AnnotatedType::Float(64)),
    ];
}
