use crate::common_ir::SRef;

use super::{EdgeWeight, MemBound, MemorizationBound, Memory};

use crate::hir::modes::{dependencies::WithDependencies, HirMode};
use crate::hir::Hir;
use std::collections::HashMap;
use std::convert::TryFrom;

pub(crate) trait MemoryAnalyzed {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound;
}

impl MemoryAnalyzed for Memory {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound {
        self.memory_bound_per_stream[&sr]
    }
}

pub(crate) trait MemoryWrapper {
    type InnerM: MemoryAnalyzed;
    fn inner_memory(&self) -> &Self::InnerM;
}

impl MemoryWrapper for MemBound {
    type InnerM = Memory;

    fn inner_memory(&self) -> &Self::InnerM {
        &self.memory
    }
}

impl<A: MemoryWrapper<InnerM = T>, T: MemoryAnalyzed + 'static> MemoryAnalyzed for A {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound {
        self.inner_memory().memory_bound(sr)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum MemoryErr {}

pub(crate) struct MemoryReport {}

type Result<T> = std::result::Result<T, MemoryErr>;

impl Memory {
    const DEFAULT_VALUE: MemorizationBound = MemorizationBound::Bounded(0);
    pub(crate) fn analyze<M>(spec: &Hir<M>) -> Result<Memory>
    where
        M: HirMode + 'static + WithDependencies,
    {
        // Assign streams to default value
        let mut memory_bounds =
            spec.all_streams().map(|sr| (sr, Self::DEFAULT_VALUE)).collect::<HashMap<SRef, MemorizationBound>>();
        // Assign stream to bounded memory
        spec.graph().edge_indices().for_each(|edge_index| {
            let cur_edge_bound = Self::edge_weight_to_memory_bound(spec.graph().edge_weight(edge_index).unwrap());
            let (src_node, _) = spec.graph().edge_endpoints(edge_index).unwrap();
            let sr = spec.graph().node_weight(src_node).unwrap();
            let cur_mem_bound = memory_bounds.get_mut(sr).unwrap();
            *cur_mem_bound = if *cur_mem_bound < cur_edge_bound { *cur_mem_bound } else { cur_edge_bound };
        });
        Ok(Memory { memory_bound_per_stream: memory_bounds })
    }

    fn edge_weight_to_memory_bound(w: &EdgeWeight) -> MemorizationBound {
        match w {
            EdgeWeight::Offset(o) => {
                if *o > 0 {
                    unimplemented!("Positive Offsets not yet implemented")
                } else {
                    MemorizationBound::Bounded(u16::try_from(*o).unwrap())
                }
            }
            EdgeWeight::Hold => Self::DEFAULT_VALUE,
            EdgeWeight::Aggr(_) => Self::DEFAULT_VALUE,
            EdgeWeight::Spawn(w) => Self::edge_weight_to_memory_bound(w),
            EdgeWeight::Filter(w) => Self::edge_weight_to_memory_bound(w),
            EdgeWeight::Close(w) => Self::edge_weight_to_memory_bound(w),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::common_ir::SRef;
    use crate::{common_ir::MemorizationBound, hir::modes::Memory};
    use std::collections::HashMap;
    #[allow(dead_code, unreachable_code, unused_variables)]
    fn check_memory_bound_for_spec(_spec: &str, ref_memory_bounds: HashMap<SRef, MemorizationBound>) {
        let bounds: Memory = todo!();
        assert_eq!(bounds.memory_bound_per_stream.len(), ref_memory_bounds.len());
        bounds.memory_bound_per_stream.iter().for_each(|(sr, b)| {
            let ref_b = ref_memory_bounds.get(sr).unwrap();
            assert_eq!(b, ref_b);
        });
    }

    #[test]
    #[ignore]
    fn synchronous_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (name_mapping["a"], MemorizationBound::Bounded(0)),
            (name_mapping["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    #[ignore]
    fn hold_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.hold().defaults(to: 0)";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (name_mapping["a"], MemorizationBound::Bounded(1)),
            (name_mapping["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    #[ignore]
    fn offset_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by: -1).defaults(to: 0)";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (name_mapping["a"], MemorizationBound::Bounded(0)),
            (name_mapping["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    #[ignore]
    fn sliding_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over: 1s, using: sum)";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (name_mapping["a"], MemorizationBound::Bounded(0)),
            (name_mapping["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    #[ignore]
    fn discrete_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.aggregate(over: 5, using: sum)";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (name_mapping["a"], MemorizationBound::Bounded(5)),
            (name_mapping["b"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }

    #[test]
    #[ignore]
    fn offset_lookups() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by:-1).defaults(to: 0)\noutput c: UInt8 := a.offset(by:-2).defaults(to: 0)\noutput d: UInt8 := a.offset(by:-3).defaults(to: 0)\noutput e: UInt8 := a.offset(by:-4).defaults(to: 0)";
        let name_mapping = vec![
            ("a", SRef::InRef(0)),
            ("b", SRef::OutRef(0)),
            ("c", SRef::OutRef(1)),
            ("d", SRef::OutRef(2)),
            ("e", SRef::OutRef(3)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (name_mapping["a"], MemorizationBound::Bounded(4)),
            (name_mapping["b"], MemorizationBound::Bounded(0)),
            (name_mapping["c"], MemorizationBound::Bounded(0)),
            (name_mapping["d"], MemorizationBound::Bounded(0)),
            (name_mapping["e"], MemorizationBound::Bounded(0)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
    #[test]
    #[ignore]
    fn negative_loop_different_offsets() {
        let spec = "input a: Int8\noutput b: Int8 := a.offset(by: -1).defaults(to: 0) + d.offset(by:-2).defaults(to:0)\noutput c: Int8 := b.offset(by:-3).defaults(to: 0)\noutput d: Int8 := c.offset(by:-4).defaults(to: 0)";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let memory_bounds = vec![
            (name_mapping["a"], MemorizationBound::Bounded(1)),
            (name_mapping["b"], MemorizationBound::Bounded(3)),
            (name_mapping["c"], MemorizationBound::Bounded(4)),
            (name_mapping["d"], MemorizationBound::Bounded(2)),
        ]
        .into_iter()
        .collect();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
}
