use crate::common_ir::{Layer, SRef, WRef};

use super::Dependencies;

pub(crate) trait DependenciesAnalyzed {
    // https://github.com/rust-lang/rust/issues/63063
    // type I1 = impl Iterator<Item = SRef>;
    // type I2 = impl Iterator<Item = (SRef, WRef)>;

    fn accesses(&self, who: SRef) -> Vec<SRef>;

    fn accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn layer(&self, sr: SRef) -> Layer;
}

impl DependenciesAnalyzed for Dependencies {
    fn accesses(&self, _who: SRef) -> Vec<SRef> {
        todo!()
    }

    fn accessed_by(&self, _who: SRef) -> Vec<SRef> {
        todo!()
    }

    fn aggregated_by(&self, _who: SRef) -> Vec<(SRef, WRef)> {
        todo!()
    }

    fn aggregates(&self, _who: SRef) -> Vec<(SRef, WRef)> {
        todo!()
    }

    fn layer(&self, _sr: SRef) -> Layer {
        todo!()
    }
}

pub(crate) trait DependenciesWrapper {
    type InnerD: DependenciesAnalyzed;
    fn inner_dep(&self) -> &Self::InnerD;
}

impl<A: DependenciesWrapper<InnerD = T>, T: DependenciesAnalyzed + 'static> DependenciesAnalyzed for A {
    fn accesses(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().accesses(who)
    }

    fn accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().accessed_by(who)
    }

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.inner_dep().aggregated_by(who)
    }

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.inner_dep().aggregates(who)
    }

    fn layer(&self, sr: SRef) -> Layer {
        self.inner_dep().layer(sr)
    }
}
