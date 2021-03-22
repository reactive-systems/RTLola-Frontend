#![allow(dead_code)]

pub mod hir;
mod modes;
pub(crate) mod stdlib;
pub mod type_check;

#[macro_use]
extern crate rtlola_macros;
