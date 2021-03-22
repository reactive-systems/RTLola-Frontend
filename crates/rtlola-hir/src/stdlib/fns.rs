//! This module contains the Lola standard library.
use crate::hir::AnnotatedType;
use crate::hir::FunctionName;
use lazy_static::lazy_static;

/// A (possibly generic) function declaration
#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: FunctionName,
    pub generics: Vec<AnnotatedType>,
    pub parameters: Vec<AnnotatedType>,
    pub return_type: AnnotatedType,
}

lazy_static! {
    // fn widen_signed<T: Signed, U: Signed>(T) -> U where U: T
    static ref WIDEN: FuncDecl = FuncDecl {
        name: FunctionName::new("widen".to_string(), &[None]),
        generics: vec![AnnotatedType::Numeric,AnnotatedType::Numeric],
        parameters: vec![AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(1, "U".to_string()),
    };
    // fn sqrt<T: FloatingPoint>(T) -> T
    static ref SQRT: FuncDecl = FuncDecl {
        name: FunctionName::new("sqrt".to_string(), &[None]),
        generics: vec![AnnotatedType::Float(0)],
        parameters: vec![AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn min<T: Numeric>(T, T) -> T
    static ref MIN: FuncDecl = FuncDecl {
        name: FunctionName::new("min".to_string(), &[None, None]),
        generics: vec![AnnotatedType::Numeric],
        parameters: vec![AnnotatedType::Param(0, "T".to_string()), AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn max<T: Numeric>(T, T) -> T
    static ref MAX: FuncDecl = FuncDecl {
        name: FunctionName::new("max".to_string(), &[None, None]),
        generics: vec![AnnotatedType::Numeric],
        parameters: vec![AnnotatedType::Param(0, "T".to_string()), AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn cos<T: FloatingPoint>(T) -> T
    static ref COS: FuncDecl = FuncDecl {
        name: FunctionName::new("cos".to_string(), &[None]),
        generics: vec![AnnotatedType::Float(0),
        ],
        parameters: vec![AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn sin<T: FloatingPoint>(T) -> T
    static ref SIN: FuncDecl = FuncDecl {
        name: FunctionName::new("sin".to_string(), &[None]),
        generics: vec![AnnotatedType::Float(0),
        ],
        parameters: vec![AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
     // fn arctan<T: FloatingPoint>(T) -> T
    static ref ARCTAN: FuncDecl = FuncDecl {
        name: FunctionName::new("arctan".to_string(), &[None]),
        generics: vec![ AnnotatedType::Float(0),
        ],
        parameters: vec![AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };
    // fn abs<T: Numeric>(T) -> T
    static ref ABS: FuncDecl = FuncDecl {
        name: FunctionName::new("abs".to_string(), &[None]),
        generics: vec![AnnotatedType::Numeric,
        ],
        parameters: vec![AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(0, "T".to_string()),
    };

    // fn matches(String, regex: String) -> Bool
    static ref MATCHES_STRING_REGEX: FuncDecl = FuncDecl {
        name: FunctionName::new("matches".to_string(), &[None, Some("regex".to_string())]),
        generics: vec![],
        parameters: vec![AnnotatedType::String, AnnotatedType::String],
        return_type: AnnotatedType::Bool,
    };

    // fn matches(Bytes, regex: String) -> Bool
    static ref MATCHES_BYTES_REGEX: FuncDecl = FuncDecl {
        name: FunctionName::new("matches".to_string(), &[None, Some("regex".to_string())]),
        generics: vec![],
        parameters: vec![AnnotatedType::Bytes, AnnotatedType::String],
        return_type: AnnotatedType::Bool,
    };

    /// fn cast<T: Numeric, U: Numeric>(T) -> U
    /// allows for arbitrary conversion of numeric types T -> U
    static ref CAST: FuncDecl = FuncDecl {
        name: FunctionName::new("cast".to_string(), &[None]),
        generics: vec![AnnotatedType::Numeric,
             AnnotatedType::Numeric,
        ],
        parameters: vec![AnnotatedType::Param(0, "T".to_string())],
        return_type: AnnotatedType::Param(1, "U".to_string()),
    };

    /// access index of byte array
    static ref BYTES_AT: FuncDecl = FuncDecl {
        name: FunctionName::new("at".to_string(), &[None, Some("index".to_string())]),
        generics: vec![],
        parameters: vec![AnnotatedType::Bytes, AnnotatedType::UInt(8)],
        return_type: AnnotatedType::Option(AnnotatedType::UInt(8).into()),
    };
}

pub(crate) fn implicit_module() -> Vec<&'static FuncDecl> {
    vec![&WIDEN, &CAST]
}

pub(crate) fn math_module() -> Vec<&'static FuncDecl> {
    vec![&SQRT, &COS, &SIN, &ABS, &ARCTAN, &MIN, &MAX]
}

pub(crate) fn regex_module() -> Vec<&'static FuncDecl> {
    vec![&MATCHES_STRING_REGEX]
}
