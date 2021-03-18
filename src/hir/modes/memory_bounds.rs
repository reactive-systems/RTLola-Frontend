use num::abs;

use crate::{common_ir::SRef, mir::StreamLayers};

use super::{
    dependencies::EdgeWeight, CompleteMode, MemBound, MemBoundMode, MemBoundTrait, MemorizationBound, OrderedMode,
};

use crate::hir::modes::{DepAnaTrait, HirMode};
use crate::hir::Hir;
use std::collections::HashMap;
use std::convert::TryFrom;

impl Hir<MemBoundMode> {
    pub(crate) fn finalize(self) -> Hir<CompleteMode> {
        let mode = CompleteMode {
            ir_expr: self.mode.ir_expr,
            dependencies: self.mode.dependencies,
            types: self.mode.types,
            layers: self.mode.layers,
            memory: self.mode.memory,
        };

        Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
        }
    }
}

impl MemBoundTrait for MemBound {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound {
        self.memory_bound_per_stream[&sr]
    }
}

pub(crate) type LayerRepresentation = HashMap<SRef, StreamLayers>;
impl Hir<OrderedMode> {
    pub(crate) fn compute_memory_bounds(self) -> Hir<MemBoundMode> {
        //TODO: forward config argument
        let memory = MemBound::analyze(&self, false);

        let mode = MemBoundMode {
            ir_expr: self.mode.ir_expr,
            dependencies: self.mode.dependencies,
            types: self.mode.types,
            layers: self.mode.layers,
            memory,
        };

        Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
        }
    }
}

impl MemBound {
    const DYNAMIC_DEFAULT_VALUE: MemorizationBound = MemorizationBound::Bounded(0);
    const STATIC_DEFAULT_VALUE: MemorizationBound = MemorizationBound::Bounded(1);
    pub(crate) fn analyze<M>(spec: &Hir<M>, dynamic: bool) -> MemBound
    where
        M: HirMode + 'static + DepAnaTrait,
    {
        // Assign streams to default value
        let mut memory_bounds = spec
            .all_streams()
            .map(|sr| (sr, if dynamic { Self::DYNAMIC_DEFAULT_VALUE } else { Self::STATIC_DEFAULT_VALUE }))
            .collect::<HashMap<SRef, MemorizationBound>>();
        // Assign stream to bounded memory
        spec.graph().edge_indices().for_each(|edge_index| {
            let cur_edge_bound =
                Self::edge_weight_to_memory_bound(spec.graph().edge_weight(edge_index).unwrap(), dynamic);
            let (_, src_node) = spec.graph().edge_endpoints(edge_index).unwrap();
            let sr = spec.graph().node_weight(src_node).unwrap();
            let cur_mem_bound = memory_bounds.get_mut(sr).unwrap();
            *cur_mem_bound = if *cur_mem_bound > cur_edge_bound { *cur_mem_bound } else { cur_edge_bound };
        });
        MemBound { memory_bound_per_stream: memory_bounds }
    }

    fn edge_weight_to_memory_bound(w: &EdgeWeight, dynamic: bool) -> MemorizationBound {
        match w {
            EdgeWeight::Offset(o) => {
                if *o > 0 {
                    unimplemented!("Positive Offsets not yet implemented")
                } else {
                    MemorizationBound::Bounded(u16::try_from(abs(*o) + if dynamic { 0 } else { 1 }).unwrap())
                }
            }
            EdgeWeight::Hold => MemorizationBound::Bounded(1),
            EdgeWeight::Aggr(_) => {
                if dynamic {
                    Self::DYNAMIC_DEFAULT_VALUE
                } else {
                    Self::STATIC_DEFAULT_VALUE
                }
            }
            EdgeWeight::Spawn(w) => Self::edge_weight_to_memory_bound(w, dynamic),
            EdgeWeight::Filter(w) => Self::edge_weight_to_memory_bound(w, dynamic),
            EdgeWeight::Close(w) => Self::edge_weight_to_memory_bound(w, dynamic),
        }
    }
}

#[cfg(test)]
mod dynaminc_memory_bound_tests {
    use super::*;
    use crate::hir::modes::IrExprMode;
    use crate::parse::parse;
    use crate::reporting::Handler;
    use crate::FrontendConfig;
    use std::path::PathBuf;
    fn check_memory_bound_for_spec(spec: &str, ref_memory_bounds: HashMap<SRef, MemorizationBound>) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<IrExprMode>::from_ast(ast, &handler, &config)
            .build_dependency_graph()
            .unwrap()
            .type_check(&handler)
            .unwrap()
            .build_evaluation_order();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
            ("a", SRef::InRef(0)),
            ("b", SRef::OutRef(0)),
            ("c", SRef::OutRef(1)),
            ("d", SRef::OutRef(2)),
            ("e", SRef::OutRef(3)),
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
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
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
            ("f", SRef::OutRef(3)),
            ("g", SRef::OutRef(4)),
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::InRef(1)), ("c", SRef::OutRef(0)), ("d", SRef::OutRef(1))]
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
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
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
    use super::*;
    use crate::hir::modes::IrExprMode;
    use crate::parse::parse;
    use crate::reporting::Handler;
    use crate::FrontendConfig;
    use std::path::PathBuf;
    fn check_memory_bound_for_spec(spec: &str, ref_memory_bounds: HashMap<SRef, MemorizationBound>) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<IrExprMode>::from_ast(ast, &handler, &config)
            .build_dependency_graph()
            .unwrap()
            .type_check(&handler)
            .unwrap()
            .build_evaluation_order();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
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
            ("a", SRef::InRef(0)),
            ("b", SRef::OutRef(0)),
            ("c", SRef::OutRef(1)),
            ("d", SRef::OutRef(2)),
            ("e", SRef::OutRef(3)),
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
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
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
            ("f", SRef::OutRef(3)),
            ("g", SRef::OutRef(4)),
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
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::InRef(1)), ("c", SRef::OutRef(0)), ("d", SRef::OutRef(1))]
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
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
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
