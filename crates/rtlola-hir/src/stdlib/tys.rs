use crate::hir::AnnotatedType;
use lazy_static::lazy_static;

// /// The `value` type, storing information about the stored values (`Bool`, `UInt8`, etc.)
// #[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
// pub enum ValueTy {
//     /// The boolean value type.
//     Bool,
//     /// A signed integer value type. See `IntTy` for more information.
//     Int(IntTy),
//     /// An unsigned integer value type. See `UIntTy` for more information.
//     UInt(UIntTy),
//     /// A floating-point value type. See `FloatTy` for more information.
//     Float(FloatTy),
//     // an abstract data type, e.g., structs, enums, etc.
//     //Adt(AdtDef),
//     /// A utf-8 encoded string type.
//     String,
//     /// A byte string type.
//     Bytes,
//     /// A tuple of value types.
//     Tuple(Vec<ValueTy>),
//     /// an optional value type, e.g., resulting from accessing a stream with offset -1
//     Option(Box<ValueTy>),
//     /// A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
//     Param(u8, String),
// }

// /**
// The possible signed integer value types.
// */
// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
// pub enum IntTy {
//     /**
//     Signed 8-bit integer value type.
//     */
//     I8,
//     /**
//     Signed 16-bit integer value type.
//     */
//     I16,
//     /**
//     Signed 32-bit integer value type.
//     */
//     I32,
//     /**
//     Signed 64-bit integer value type.
//     */
//     I64,
// }
// use crate::hir::AnnotatedType;

// use self::IntTy::*;

// /**
// The possible unsigned integer value types.
// */
// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
// pub enum UIntTy {
//     /**
//     Unsigned 8-bit integer value type.
//     */
//     U8,
//     /**
//     Unsigned 16-bit integer value type.
//     */
//     U16,
//     /**
//     Unsigned 32-bit integer value type.
//     */
//     U32,
//     /**
//     Unsigned 64-bit integer value type.
//     */
//     U64,
// }
// use self::UIntTy::*;

// /**
// The possible floating-point value types.
// */
// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
// pub enum FloatTy {
//     /**
//     16-bit floating-point value type.
//     */
//     F16,
//     /**
//     32-bit floating-point value type.
//     */
//     F32,
//     /**
//     64-bit floating-point value type.
//     */
//     F64,
// }
// use self::FloatTy::*;

// /**
// The activation condition describes when an event-based stream produces a new value.
// */
// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
// pub enum Activation<Var> {
//     /**
//     When all of the activation conditions is true.
//     */
//     Conjunction(Vec<Self>),
//     /**
//     When one of the activation conditions is true.
//     */
//     Disjunction(Vec<Self>),
//     /**
//     Whenever the specified stream produces a new value.
//     */
//     Stream(Var),
//     /**
//     Whenever an event-based stream produces a new value.
//     */
//     True,
// }

// impl std::fmt::Display for ValueTy {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         match self {
//             ValueTy::Bool => write!(f, "Bool"),
//             ValueTy::Int(I8) => write!(f, "Int8"),
//             ValueTy::Int(I16) => write!(f, "Int16"),
//             ValueTy::Int(I32) => write!(f, "Int32"),
//             ValueTy::Int(I64) => write!(f, "Int64"),
//             ValueTy::UInt(U8) => write!(f, "UInt8"),
//             ValueTy::UInt(U16) => write!(f, "UInt16"),
//             ValueTy::UInt(U32) => write!(f, "UInt32"),
//             ValueTy::UInt(U64) => write!(f, "UInt64"),
//             ValueTy::Float(F16) => write!(f, "Float16"),
//             ValueTy::Float(F32) => write!(f, "Float32"),
//             ValueTy::Float(F64) => write!(f, "Float64"),
//             ValueTy::String => write!(f, "String"),
//             ValueTy::Bytes => write!(f, "Bytes"),
//             ValueTy::Option(ty) => write!(f, "{}?", ty),
//             ValueTy::Tuple(inner) => {
//                 let joined: Vec<String> = inner.iter().map(|e| format!("{}", e)).collect();
//                 write!(f, "({})", joined.join(", "))
//             }
//             ValueTy::Param(_, name) => write!(f, "{}", name),
//         }
//     }
// }

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Copy)]
// pub enum TypeConstraint {
//     /**
//     The type must be a signed integer type.
//     */
//     SignedInteger,
//     /**
//     The type must be an unsigned integer type.
//     */
//     UnsignedInteger,
//     /**
//     The type must be a floating-point type.
//     */
//     FloatingPoint,
//     /// signed + unsigned integer
//     Integer,
//     /// integer + floating point
//     Numeric,
//     /// Types that can be compared, i.e., implement `==`
//     Equatable,
//     /// Types that can be ordered, i.e., implement `<`, `>`,
//     Comparable,
//     /**
//     The type is unconstrained.
//     */
//     Unconstrained,
// }

// impl std::fmt::Display for TypeConstraint {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         use self::TypeConstraint::*;
//         match self {
//             SignedInteger => write!(f, "signed integer"),
//             UnsignedInteger => write!(f, "unsigned integer"),
//             Integer => write!(f, "integer"),
//             FloatingPoint => write!(f, "floating point"),
//             Numeric => write!(f, "numeric type"),
//             Equatable => write!(f, "equatable type"),
//             Comparable => write!(f, "comparable type"),
//             Unconstrained => write!(f, "unconstrained type"),
//         }
//     }
// }
