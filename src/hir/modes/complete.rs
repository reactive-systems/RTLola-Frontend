use crate::{hir::Hir, mir::Mir};

use super::Complete;

impl Hir<Complete> {
    pub(crate) fn lower(self) -> Mir {
        unimplemented!()
    }
}
