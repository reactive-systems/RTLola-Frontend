//! Contains tag validators accepting all tags or no tags at all

use std::collections::{HashMap, HashSet};

use rtlola_reporting::RtLolaError;

use super::{TagParser, TagValidator};
use crate::{mir::StreamReference, RtLolaMir};

#[derive(Debug, Copy, Clone)]
/// A tag validator allowing any tags in the specification
pub struct AllowAll;

impl TagParser for AllowAll {
    type GlobalTags = ();
    type LocalTags = ();

    fn parse_global(
        &self,
        _global_tags: &HashMap<String, Option<String>>,
        _mir: &RtLolaMir,
    ) -> Result<Self::GlobalTags, RtLolaError> {
        Ok(())
    }

    fn parse_local(
        &self,
        _sr: StreamReference,
        _tags: &HashMap<String, Option<String>>,
        _mir: &RtLolaMir,
    ) -> Result<Self::LocalTags, RtLolaError> {
        Ok(())
    }
}

impl TagValidator for AllowAll {
    fn supported_tags<'a>(&self, mir: &'a RtLolaMir) -> (HashSet<&'a str>, HashSet<&'a str>) {
        (mir.all_global_tags(), mir.all_local_tags())
    }
}

#[derive(Debug, Copy, Clone)]
/// A tag validator allowing no tags in the specification
pub struct DenyAll;

impl TagParser for DenyAll {
    type GlobalTags = ();
    type LocalTags = ();

    fn parse_global(
        &self,
        _global_tags: &HashMap<String, Option<String>>,
        _mir: &RtLolaMir,
    ) -> Result<Self::GlobalTags, RtLolaError> {
        Ok(())
    }

    fn parse_local(
        &self,
        _sr: StreamReference,
        _tags: &HashMap<String, Option<String>>,
        _mir: &RtLolaMir,
    ) -> Result<Self::LocalTags, RtLolaError> {
        Ok(())
    }
}

impl TagValidator for DenyAll {
    fn supported_tags<'a>(&self, _mir: &'a RtLolaMir) -> (HashSet<&'a str>, HashSet<&'a str>) {
        (HashSet::new(), HashSet::new())
    }
}
