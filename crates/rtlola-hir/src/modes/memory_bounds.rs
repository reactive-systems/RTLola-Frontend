use std::collections::HashMap;
use std::convert::TryFrom;

use num::abs;

use crate::hir::{Hir, SRef};
use crate::modes::dependencies::EdgeWeight;
use crate::modes::{DepAnaTrait, HirMode, MemBound, MemBoundTrait};

/// This enum indicates how much memory is required to store a stream.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MemorizationBound {
    /// The required memory might exceed any bound.
    Unbounded,
    /// Only the contained amount of stream entries need to be stored.
    Bounded(u16),
}

impl MemorizationBound {
    /// Returns the unwraped memory bound
    ///
    /// This function returns the bound of an [MemorizationBound::Bounded] as `u16`. The function panics, if it is called on a [MemorizationBound::Unbounded] value.
    pub fn unwrap(self) -> u16 {
        match self {
            MemorizationBound::Bounded(b) => b,
            MemorizationBound::Unbounded => {
                unreachable!("Called `MemorizationBound::unwrap()` on an `Unbounded` value.")
            },
        }
    }

    /// Returns the unwraped memory bound as an optional
    ///
    /// This function returns `Some(v)` if the memory is bounded and `v` and `None` if it is unbounded.
    pub fn as_opt(self) -> Option<u16> {
        match self {
            MemorizationBound::Bounded(b) => Some(b),
            MemorizationBound::Unbounded => None,
        }
    }
}

/// Represents the error of the memory bound analysis
#[derive(Debug, Clone, Copy)]
pub enum MemBoundErr {}

impl PartialOrd for MemorizationBound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        use MemorizationBound::*;
        match (self, other) {
            (Unbounded, Unbounded) => None,
            (Bounded(_), Unbounded) => Some(Ordering::Less),
            (Unbounded, Bounded(_)) => Some(Ordering::Greater),
            (Bounded(b1), Bounded(b2)) => Some(b1.cmp(&b2)),
        }
    }
}

impl MemBoundTrait for MemBound {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound {
        self.memory_bound_per_stream[&sr]
    }
}

impl MemBound {
    const DYNAMIC_DEFAULT_VALUE: MemorizationBound = MemorizationBound::Bounded(0);
    const STATIC_DEFAULT_VALUE: MemorizationBound = MemorizationBound::Bounded(1);

    /// Returns the result of the memory analysis
    ///
    /// This function returns for each stream in `spec` the required memory. It differentiates with the `dynamic` flag between a dynamic and a static memory computation.
    /// The dynamic memory computation starts with a memory-bound of 0 and increases the bound only if a value is used in at least one other evaluation cycle, i.e., if synchronous lookups only access a stream with an offset of 0, then this value does not need to be store in the global memory and the bound for this stream is 0.
    /// The static memory computation assumes that each value is stored in the global memory, so the starting value of each stream is 1.
    pub(crate) fn analyze<M>(spec: &Hir<M>, dynamic: bool) -> MemBound
    where
        M: HirMode + DepAnaTrait,
    {
        // Assign streams to default value
        let mut memory_bounds = spec
            .all_streams()
            .map(|sr| {
                (
                    sr,
                    if dynamic {
                        Self::DYNAMIC_DEFAULT_VALUE
                    } else {
                        Self::STATIC_DEFAULT_VALUE
                    },
                )
            })
            .collect::<HashMap<SRef, MemorizationBound>>();
        // Assign stream to bounded memory
        spec.graph().edge_indices().for_each(|edge_index| {
            let cur_edge_bound =
                Self::edge_weight_to_memory_bound(spec.graph().edge_weight(edge_index).unwrap(), dynamic);
            let (_, src_node) = spec.graph().edge_endpoints(edge_index).unwrap();
            let sr = spec.graph().node_weight(src_node).unwrap();
            let cur_mem_bound = memory_bounds.get_mut(sr).unwrap();
            *cur_mem_bound = if *cur_mem_bound > cur_edge_bound {
                *cur_mem_bound
            } else {
                cur_edge_bound
            };
        });
        MemBound {
            memory_bound_per_stream: memory_bounds,
        }
    }

    fn edge_weight_to_memory_bound(w: &EdgeWeight, dynamic: bool) -> MemorizationBound {
        match w {
            EdgeWeight::Offset(o) => {
                if *o > 0 {
                    unimplemented!("Positive Offsets not yet implemented")
                } else {
                    MemorizationBound::Bounded(u16::try_from(abs(*o) + if dynamic { 0 } else { 1 }).unwrap())
                }
            },
            EdgeWeight::Hold => MemorizationBound::Bounded(1),
            EdgeWeight::Aggr(_) => {
                if dynamic {
                    Self::DYNAMIC_DEFAULT_VALUE
                } else {
                    Self::STATIC_DEFAULT_VALUE
                }
            },
            EdgeWeight::Spawn(w) => Self::edge_weight_to_memory_bound(w, dynamic),
            EdgeWeight::Filter(w) => Self::edge_weight_to_memory_bound(w, dynamic),
            EdgeWeight::Close(w) => Self::edge_weight_to_memory_bound(w, dynamic),
        }
    }
}

#[cfg(test)]
mod dynaminc_memory_bound_tests {
    use std::path::PathBuf;

    use rtlola_parser::{parse_with_handler, ParserConfig};
    use rtlola_reporting::Handler;

    use super::*;
    use crate::modes::BaseMode;
    fn check_memory_bound_for_spec(spec: &str, ref_memory_bounds: HashMap<SRef, MemorizationBound>) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let ast = parse_with_handler(ParserConfig::for_string(spec.to_string()), &handler)
            .unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<BaseMode>::from_ast(ast, &handler)
            .unwrap()
            .analyze_dependencies(&handler)
            .unwrap()
            .check_types(&handler)
            .unwrap()
            .determine_evaluation_order(&handler)
            .unwrap();
        let bounds = MemBound::analyze(&hir, true);
        assert_eq!(bounds.memory_bound_per_stream.len(), ref_memory_bounds.len());
        bounds.memory_bound_per_stream.iter().for_each(|(sr, b)| {
            let ref_b = ref_memory_bounds.get(sr).unwrap();
            assert_eq!(b, ref_b);
        });
    }

    #[test]
    fn synchronous_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(0)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn hold_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.hold().defaults(to: 0)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn offset_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by: -1).defaults(to: 0)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn sliding_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over: 1s, using: sum)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(0)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn discrete_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over_discrete: 5, using: sum)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(0)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn offset_lookups() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by:-1).defaults(to: 0)\noutput c: UInt8 := a.offset(by:-2).defaults(to: 0)\noutput d: UInt8 := a.offset(by:-3).defaults(to: 0)\noutput e: UInt8 := a.offset(by:-4).defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
            ("e", SRef::Out(3)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(4)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
            (sname_to_sref["c"], MemorizationBound::Bounded(0)),
            (sname_to_sref["d"], MemorizationBound::Bounded(0)),
            (sname_to_sref["e"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
    #[test]
    fn negative_loop_different_offsets() {
        let spec = "input a: Int8\noutput b: Int8 := a.offset(by: -1).defaults(to: 0) + d.offset(by:-2).defaults(to:0)\noutput c: Int8 := b.offset(by:-3).defaults(to: 0)\noutput d: Int8 := c.offset(by:-4).defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(3)),
            (sname_to_sref["c"], MemorizationBound::Bounded(4)),
            (sname_to_sref["d"], MemorizationBound::Bounded(2)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn parameter_loop_with_lookup_in_close() {
        let spec = "input a: Int8\ninput b: Int8\noutput c(p) spawn with a if a < b := p + b + g(p).hold().defaults(to: 0)\noutput d(p) spawn with b if c(4).hold().defaults(to: 0) < 10 := b + 5\noutput e(p)@b spawn with b := d(p).hold().defaults(to: 0) + 5\noutput f(p) spawn with b filter e(p).hold().defaults(to: 0) < 6 := b + 5\noutput g(p) spawn with b close f(p).hold().defaults(to: 0) < 6 := b + 5";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
            ("f", SRef::Out(3)),
            ("g", SRef::Out(4)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(0)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
            (sname_to_sref["c"], MemorizationBound::Bounded(1)),
            (sname_to_sref["d"], MemorizationBound::Bounded(1)),
            (sname_to_sref["e"], MemorizationBound::Bounded(1)),
            (sname_to_sref["f"], MemorizationBound::Bounded(1)),
            (sname_to_sref["g"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn parameter_nested_lookup_implicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(c(b).hold().defaults(to: 0)).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(0)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
            (sname_to_sref["c"], MemorizationBound::Bounded(1)),
            (sname_to_sref["d"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
    #[test]
    fn parameter_nested_lookup_explicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(0)),
            (sname_to_sref["b"], MemorizationBound::Bounded(0)),
            (sname_to_sref["c"], MemorizationBound::Bounded(1)),
            (sname_to_sref["d"], MemorizationBound::Bounded(0)),
            (sname_to_sref["e"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
}

#[cfg(test)]
mod static_memory_bound_tests {
    use std::path::PathBuf;

    use rtlola_parser::{parse_with_handler, ParserConfig};
    use rtlola_reporting::Handler;

    use super::*;
    use crate::modes::BaseMode;
    fn check_memory_bound_for_spec(spec: &str, ref_memory_bounds: HashMap<SRef, MemorizationBound>) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let ast = parse_with_handler(ParserConfig::for_string(spec.to_string()), &handler)
            .unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<BaseMode>::from_ast(ast, &handler)
            .unwrap()
            .analyze_dependencies(&handler)
            .unwrap()
            .check_types(&handler)
            .unwrap()
            .determine_evaluation_order(&handler)
            .unwrap();
        let bounds = MemBound::analyze(&hir, false);
        assert_eq!(bounds.memory_bound_per_stream.len(), ref_memory_bounds.len());
        bounds.memory_bound_per_stream.iter().for_each(|(sr, b)| {
            let ref_b = ref_memory_bounds.get(sr).unwrap();
            assert_eq!(b, ref_b);
        });
    }

    #[test]
    fn synchronous_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn hold_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.hold().defaults(to: 0)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn offset_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by: -1).defaults(to: 0)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(2)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn discrete_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over_discrete: 5, using: sum)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn sliding_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over: 1s, using: sum)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn offset_lookups() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by:-1).defaults(to: 0)\noutput c: UInt8 := a.offset(by:-2).defaults(to: 0)\noutput d: UInt8 := a.offset(by:-3).defaults(to: 0)\noutput e: UInt8 := a.offset(by:-4).defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
            ("e", SRef::Out(3)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(5)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
            (sname_to_sref["c"], MemorizationBound::Bounded(1)),
            (sname_to_sref["d"], MemorizationBound::Bounded(1)),
            (sname_to_sref["e"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
    #[test]
    fn negative_loop_different_offsets() {
        let spec = "input a: Int8\noutput b: Int8 := a.offset(by: -1).defaults(to: 0) + d.offset(by:-2).defaults(to:0)\noutput c: Int8 := b.offset(by:-3).defaults(to: 0)\noutput d: Int8 := c.offset(by:-4).defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(2)),
            (sname_to_sref["b"], MemorizationBound::Bounded(4)),
            (sname_to_sref["c"], MemorizationBound::Bounded(5)),
            (sname_to_sref["d"], MemorizationBound::Bounded(3)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn parameter_loop_with_lookup_in_close() {
        let spec = "input a: Int8\ninput b: Int8\noutput c(p) spawn with a if a < b := p + b + g(p).hold().defaults(to: 0)\noutput d(p) spawn with b if c(4).hold().defaults(to: 0) < 10 := b + 5\noutput e(p)@b spawn with b := d(p).hold().defaults(to: 0) + 5\noutput f(p) spawn with b filter e(p).hold().defaults(to: 0) < 6 := b + 5\noutput g(p) spawn with b close f(p).hold().defaults(to: 0) < 6 := b + 5";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
            ("f", SRef::Out(3)),
            ("g", SRef::Out(4)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
            (sname_to_sref["c"], MemorizationBound::Bounded(1)),
            (sname_to_sref["d"], MemorizationBound::Bounded(1)),
            (sname_to_sref["e"], MemorizationBound::Bounded(1)),
            (sname_to_sref["f"], MemorizationBound::Bounded(1)),
            (sname_to_sref["g"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    fn parameter_nested_lookup_implicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(c(b).hold().defaults(to: 0)).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
            (sname_to_sref["c"], MemorizationBound::Bounded(1)),
            (sname_to_sref["d"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
    #[test]
    fn parameter_nested_lookup_explicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (sname_to_sref["a"], MemorizationBound::Bounded(1)),
            (sname_to_sref["b"], MemorizationBound::Bounded(1)),
            (sname_to_sref["c"], MemorizationBound::Bounded(1)),
            (sname_to_sref["d"], MemorizationBound::Bounded(1)),
            (sname_to_sref["e"], MemorizationBound::Bounded(1)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
}
