use std::collections::HashMap;
use std::ops::Add;
use std::time::Duration;

use num::integer::{gcd, lcm};
use num::traits::Inv;
use serde::{Deserialize, Serialize};
use uom::si::rational64::Time as UOM_Time;

use super::dependencies::Origin;
use super::TypedTrait;
use crate::hir::{ConcretePacingType, Hir, SRef, StreamAccessKind, WRef, WindowReference};
use crate::modes::{DepAnaTrait, HirMode, MemBound, MemBoundTrait};

/// This enum indicates how much memory is required to store a stream.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum MemorizationBound {
    /// The required memory might exceed any bound.
    Unbounded,
    /// Only the contained amount of stream entries need to be stored.
    Bounded(u32),
}

impl MemorizationBound {
    const DYNAMIC_DEFAULT_VALUE: MemorizationBound = MemorizationBound::Bounded(0);
    const STATIC_DEFAULT_VALUE: MemorizationBound = MemorizationBound::Bounded(1);

    /// Returns the unwraped memory bound
    ///
    /// This function returns the bound of an [MemorizationBound::Bounded] as `u16`. The function panics, if it is called on a [MemorizationBound::Unbounded] value.
    pub fn unwrap(self) -> u32 {
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
    pub fn as_opt(self) -> Option<u32> {
        match self {
            MemorizationBound::Bounded(b) => Some(b),
            MemorizationBound::Unbounded => None,
        }
    }

    /// Returns the default value for the [MemorizationBound]
    pub(crate) fn default_value(dynamic: bool) -> MemorizationBound {
        if dynamic {
            Self::DYNAMIC_DEFAULT_VALUE
        } else {
            Self::STATIC_DEFAULT_VALUE
        }
    }
}

impl Add for MemorizationBound {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MemorizationBound::Unbounded, MemorizationBound::Unbounded)
            | (MemorizationBound::Unbounded, MemorizationBound::Bounded(_))
            | (MemorizationBound::Bounded(_), MemorizationBound::Unbounded) => MemorizationBound::Unbounded,
            (MemorizationBound::Bounded(lhs), MemorizationBound::Bounded(rhs)) => MemorizationBound::Bounded(lhs + rhs),
        }
    }
}

impl PartialOrd for MemorizationBound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        use MemorizationBound::*;
        match (self, other) {
            (Unbounded, Unbounded) => None,
            (Bounded(_), Unbounded) => Some(Ordering::Less),
            (Unbounded, Bounded(_)) => Some(Ordering::Greater),
            (Bounded(b1), Bounded(b2)) => Some(b1.cmp(b2)),
        }
    }
}

impl MemBoundTrait for MemBound {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound {
        self.memory_bound_per_stream[&sr]
    }

    fn num_buckets(&self, wr: WRef) -> MemorizationBound {
        self.memory_bound_per_window[&wr]
    }

    fn bucket_size(&self, wr: WRef) -> Duration {
        assert!(matches!(wr, WindowReference::Sliding(_)));
        self.sliding_window_bucket_size[&wr]
    }
}

impl MemBound {
    /// Returns the result of the memory analysis
    ///
    /// This function returns for each stream in `spec` the required memory. It differentiates with the `dynamic` flag between a dynamic and a static memory computation.
    /// The dynamic memory computation starts with a memory-bound of 0 and increases the bound only if a value is used in at least one other evaluation cycle, i.e., if synchronous lookups only access a stream with an offset of 0, then this value does not need to be store in the global memory and the bound for this stream is 0.
    /// The static memory computation assumes that each value is stored in the global memory, so the starting value of each stream is 1.
    pub(crate) fn analyze<M>(spec: &Hir<M>, dynamic: bool) -> MemBound
    where
        M: HirMode + DepAnaTrait + TypedTrait,
    {
        // Assign streams to default value
        let mut memory_bound_per_stream = spec
            .all_streams()
            .map(|sr| (sr, MemorizationBound::default_value(dynamic)))
            .collect::<HashMap<SRef, MemorizationBound>>();

        let mut memory_bound_per_window = HashMap::new();
        let mut sliding_window_bucket_size = HashMap::new();

        // Assign stream to bounded memory
        spec.graph().edge_indices().for_each(|edge_index| {
            let cur_edge_weight = spec.graph().edge_weight(edge_index).unwrap();
            let cur_edge_bound = cur_edge_weight.as_memory_bound(dynamic);
            let (_, src_node) = spec.graph().edge_endpoints(edge_index).unwrap();
            let sr = spec.graph().node_weight(src_node).unwrap();
            let cur_mem_bound = memory_bound_per_stream.get_mut(sr).unwrap();
            *cur_mem_bound = if *cur_mem_bound > cur_edge_bound {
                *cur_mem_bound
            } else {
                cur_edge_bound
            };

            if let StreamAccessKind::SlidingWindow(wr) | StreamAccessKind::DiscreteWindow(wr) = cur_edge_weight.kind {
                let num_buckets = match wr {
                    WindowReference::Sliding(_) => {
                        let (bucket_count, bucket_size) =
                            Self::calculate_num_window_buckets(spec, wr, cur_edge_weight.origin);
                        assert!(!sliding_window_bucket_size.contains_key(&wr));
                        sliding_window_bucket_size.insert(wr, bucket_size);
                        bucket_count
                    },
                    WindowReference::Discrete(_) => spec.single_discrete(wr).aggr.duration,
                    WindowReference::Instance(_) => unreachable!(),
                };
                let memory_bound = MemorizationBound::Bounded(num_buckets as u32);
                assert!(!memory_bound_per_window.contains_key(&wr));
                memory_bound_per_window.insert(wr, memory_bound);
            }
        });

        MemBound {
            memory_bound_per_stream,
            memory_bound_per_window,
            sliding_window_bucket_size,
        }
    }

    fn calculate_num_window_buckets<M>(spec: &Hir<M>, window: WRef, origin: Origin) -> (usize, Duration)
    where
        M: HirMode + TypedTrait,
    {
        let window = &spec.single_sliding(window);
        let caller_ty = spec.mode.stream_type(window.caller);

        let caller_pacing = match origin {
            Origin::Spawn => caller_ty.spawn_pacing,
            Origin::Filter(_) | Origin::Eval(_) => caller_ty.eval_pacing,
            Origin::Close => caller_ty.close_pacing,
        };

        let caller_frequency = match caller_pacing {
            ConcretePacingType::FixedGlobalPeriodic(p) | ConcretePacingType::FixedLocalPeriodic(p) => p,
            _ => panic!("windows can only aggregate periodic streams with fixed frequency",),
        };
        let caller_period =
            UOM_Time::new::<uom::si::time::second>(caller_frequency.get::<uom::si::frequency::hertz>().inv())
                .get::<uom::si::time::nanosecond>()
                .to_integer();

        let window_duration = window.aggr.duration.as_nanos() as i64;
        let bucket_count = (lcm(window_duration, caller_period) / caller_period) as usize;
        let bucket_size = gcd(window_duration, caller_period);
        let bucket_size = Duration::from_nanos(bucket_size as u64);

        (bucket_count, bucket_size)
    }
}

#[cfg(test)]
mod dynaminc_memory_bound_tests {
    use rtlola_parser::{parse, ParserConfig};

    use super::*;
    use crate::modes::BaseMode;
    fn check_memory_bound_for_spec(spec: &str, ref_memory_bounds: HashMap<SRef, MemorizationBound>) {
        let ast = parse(&ParserConfig::for_string(spec.to_string())).unwrap_or_else(|e| panic!("{:?}", e));
        let hir = Hir::<BaseMode>::from_ast(ast)
            .unwrap()
            .check_types()
            .unwrap()
            .analyze_dependencies()
            .unwrap()
            .determine_evaluation_order()
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
        let spec = "input a: UInt8\noutput b: UInt8 := a.aggregate(over_discrete: 5, using: sum)";
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
        let spec = "input a: Int8\ninput b: Int8\noutput c(p) spawn with a when a < b eval with p + b + g(p).hold().defaults(to: 0)\noutput d(p) spawn with b when c(4).hold().defaults(to: 0) < 10 eval with b + 5\noutput e(p) spawn with b eval @b with d(p).hold().defaults(to: 0) + 5\noutput f(p) spawn with b eval when e(p).hold().defaults(to: 0) < 6 with b + 5\noutput g(p) spawn with b close @true when f(p).hold().defaults(to: 0) < 6 eval with b + 5";
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
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a eval with p + b\noutput d := c(c(b).hold().defaults(to: 0)).hold().defaults(to: 0)";
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
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a eval with p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
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
    use rtlola_parser::{parse, ParserConfig};

    use super::*;
    use crate::modes::BaseMode;

    fn calculate_memory_bound(spec: &str) -> MemBound {
        let ast = parse(&ParserConfig::for_string(spec.to_string())).unwrap_or_else(|e| panic!("{:?}", e));
        let hir = Hir::<BaseMode>::from_ast(ast)
            .unwrap()
            .check_types()
            .unwrap()
            .analyze_dependencies()
            .unwrap()
            .determine_evaluation_order()
            .unwrap();
        MemBound::analyze(&hir, false)
    }
    fn check_memory_bound_for_spec(spec: &str, ref_memory_bounds: HashMap<SRef, MemorizationBound>) {
        let bounds = calculate_memory_bound(spec);
        assert_eq!(bounds.memory_bound_per_stream.len(), ref_memory_bounds.len());
        bounds.memory_bound_per_stream.iter().for_each(|(sr, b)| {
            let ref_b = ref_memory_bounds.get(sr).unwrap();
            assert_eq!(b, ref_b);
        });
    }

    fn check_memory_bound_for_windows(spec: &str, ref_memory_bounds: HashMap<WRef, (MemorizationBound, Duration)>) {
        let bounds = calculate_memory_bound(spec);
        assert_eq!(bounds.memory_bound_per_window.len(), ref_memory_bounds.len());
        bounds.memory_bound_per_window.iter().for_each(|(wr, b)| {
            let ref_b = ref_memory_bounds.get(wr).unwrap().0.unwrap();
            assert_eq!(b.unwrap(), ref_b, "{}", wr);
        });
        bounds.sliding_window_bucket_size.iter().for_each(|(wr, b)| {
            let ref_b = ref_memory_bounds.get(wr).unwrap().1;
            assert_eq!(*b, ref_b, "{}", wr);
        })
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
        let spec = "input a: UInt8\noutput b: UInt8 := a.aggregate(over_discrete: 5, using: sum)";
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
        let spec = "input a: Int8\ninput b: Int8\noutput c(p) spawn with a when a < b eval with p + b + g(p).hold().defaults(to: 0)\noutput d(p) spawn with b when c(4).hold().defaults(to: 0) < 10 eval with b + 5\noutput e(p) spawn with b eval @b with d(p).hold().defaults(to: 0) + 5\noutput f(p) spawn with b eval when e(p).hold().defaults(to: 0) < 6 with b + 5\noutput g(p) spawn with b close @true when f(p).hold().defaults(to: 0) < 6 eval with b + 5";
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
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a eval with p + b\noutput d := c(c(b).hold().defaults(to: 0)).hold().defaults(to: 0)";
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
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a eval with p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
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

    #[test]
    fn sliding_window_memory_bound() {
        let spec = "input a : UInt64\noutput b@1s := a.aggregate(over: 3s, using: sum)\noutput c@4s := a.aggregate(over: 2s, using: sum)\noutput d@2s := a.aggregate(over: 3s, using: sum)\noutput e@3s := a.aggregate(over: 2s, using: sum)";
        let ref_memory_bounds = vec![
            (
                WRef::Sliding(0),
                (MemorizationBound::Bounded(3), Duration::from_secs(1)),
            ),
            (
                WRef::Sliding(1),
                (MemorizationBound::Bounded(1), Duration::from_secs(2)),
            ),
            (
                WRef::Sliding(2),
                (MemorizationBound::Bounded(3), Duration::from_secs(1)),
            ),
            (
                WRef::Sliding(3),
                (MemorizationBound::Bounded(2), Duration::from_secs(1)),
            ),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_windows(spec, ref_memory_bounds)
    }
}
