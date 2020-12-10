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
    use crate::common_ir::MemorizationBound;
    use crate::common_ir::SRef;
    use std::collections::HashMap;
    fn check_memory_bound_for_spec(_spec: &str, _memory_bounds: HashMap<SRef, MemorizationBound>) {
        todo!()
    }

    #[test]
    #[ignore]
    fn simple_spec() {
        let spec = "input a: UInt8\noutput b: UInt8 := a";
        let memory_bounds = todo!();
        check_memory_bound_for_spec(spec, memory_bounds)
    }
}
