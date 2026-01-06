use rtlola_reporting::RtLolaError;

use crate::hir::{DepAnaMode, Hir};

impl Hir<DepAnaMode> {
    pub(crate) fn add_privacy_barriers(self, parameters: f64) -> Result<Self, RtLolaError> {
        todo!()
    }
}
