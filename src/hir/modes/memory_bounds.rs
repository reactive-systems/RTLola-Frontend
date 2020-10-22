use crate::common_ir::SRef;

use super::{MemorizationBound, Memory};

use crate::hir::modes::{dependencies::DependenciesAnalyzed, HirMode};
use crate::hir::Hir;

pub(crate) trait MemoryAnalyzed {
    fn memory(&self, sr: SRef) -> MemorizationBound;
}

impl MemoryAnalyzed for Memory {
    fn memory(&self, sr: SRef) -> MemorizationBound {
        self.memory_bound_per_stram[&sr]
    }
}

pub(crate) trait MemoryWrapper {
    type InnerM: MemoryAnalyzed;
    fn inner_memory(&self) -> &Self::InnerM;
}

impl<A: MemoryWrapper<InnerM = T>, T: MemoryAnalyzed + 'static> MemoryAnalyzed for A {
    fn memory(&self, sr: SRef) -> MemorizationBound {
        self.inner_memory().memory(sr)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum MemoryErr {}

pub(crate) struct MemoryReport {}

type Result<T> = std::result::Result<T, MemoryErr>;

impl Memory {
    pub(crate) fn analyze<M>(_spec: &Hir<M>) -> Result<Memory>
    where
        M: HirMode + 'static + DependenciesAnalyzed,
    {
        todo!()
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
