use crate::common_ir::{SRef, WRef};

use super::{Dependencies, EdgeWeight};

use std::collections::HashMap;
use std::convert::TryFrom;

use super::dg_functionality::*;
use crate::hir::expression::{Expression, ExpressionKind};
use crate::hir::{expression::StreamAccessKind, Hir};
use crate::{
    common_ir::Offset,
    hir::modes::{ir_expr::WithIrExpr, HirMode},
};
use petgraph::{algo::is_cyclic_directed, graph::NodeIndex, Graph};

pub(crate) trait WithDependencies {
    // https://github.com/rust-lang/rust/issues/63063
    // type I1 = impl Iterator<Item = SRef>;
    // type I2 = impl Iterator<Item = (SRef, WRef)>;

    fn accesses(&self, who: SRef) -> Vec<SRef>;

    fn accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn graph(&self) -> &Graph<SRef, EdgeWeight>;
}

impl WithDependencies for Dependencies {
    fn accesses(&self, who: SRef) -> Vec<SRef> {
        self.accesses.get(&who).map_or(Vec::new(), |accesses| accesses.iter().copied().collect::<Vec<SRef>>())
    }

    fn accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.accessed_by.get(&who).map_or(Vec::new(), |accessed_by| accessed_by.iter().copied().collect::<Vec<SRef>>())
    }

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.aggregated_by.get(&who).map_or(Vec::new(), |aggregated_by| {
            aggregated_by.iter().map(|(sref, wref)| (*sref, *wref)).collect::<Vec<(SRef, WRef)>>()
        })
    }

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.aggregates.get(&who).map_or(Vec::new(), |aggregates| {
            aggregates.iter().map(|(sref, wref)| (*sref, *wref)).collect::<Vec<(SRef, WRef)>>()
        })
    }

    fn graph(&self) -> &Graph<SRef, EdgeWeight> {
        &self.graph
    }
}

pub(crate) trait DependenciesWrapper {
    type InnerD: WithDependencies;
    fn inner_dep(&self) -> &Self::InnerD;
}

impl<A: DependenciesWrapper<InnerD = T>, T: WithDependencies + 'static> WithDependencies for A {
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

    fn graph(&self) -> &Graph<SRef, EdgeWeight> {
        self.inner_dep().graph()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DependencyErr {
    NegativeCycle, // Should probably contain the cycle
    WellFormedNess,
}

pub(crate) struct DependencyReport {}

type Result<T> = std::result::Result<T, DependencyErr>;

impl Dependencies {
    pub(crate) fn analyze<M>(spec: &Hir<M>) -> Result<Dependencies>
    where
        M: WithIrExpr + HirMode + 'static,
    {
        let num_nodes = spec.num_inputs() + spec.num_outputs() + spec.num_triggers();
        let num_edges = num_nodes; // Todo: improve estimate.
        let mut graph: Graph<SRef, EdgeWeight> = Graph::with_capacity(num_nodes, num_edges);
        let edges_expr = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| Self::collect_edges(spec, sr, spec.expr(sr)))
            .map(|(src, w, tar)| (src, Self::stream_access_kind_to_edge_weight(w), tar));
        let edges_spawn = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| {
                spec.spawn(sr).map(|(spawn_expr, spawn_cond)| {
                    Self::collect_edges(spec, sr, spawn_expr)
                        .into_iter()
                        .chain(spawn_cond.map_or(Vec::new(), |spawn_cond| Self::collect_edges(spec, sr, spawn_cond)))
                })
            })
            .flatten()
            .map(|(src, w, tar)| (src, EdgeWeight::Spawn(Box::new(Self::stream_access_kind_to_edge_weight(w))), tar));
        let edges_filter = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| spec.filter(sr).map(|filter| Self::collect_edges(spec, sr, filter)))
            .flatten()
            .map(|(src, w, tar)| (src, EdgeWeight::Filter(Box::new(Self::stream_access_kind_to_edge_weight(w))), tar));
        let edges_close = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| spec.close(sr).map(|close| Self::collect_edges(spec, sr, close)))
            .flatten()
            .map(|(src, w, tar)| (src, EdgeWeight::Close(Box::new(Self::stream_access_kind_to_edge_weight(w))), tar));
        let edges = edges_expr
            .chain(edges_spawn)
            .chain(edges_filter)
            .chain(edges_close)
            .collect::<Vec<(SRef, EdgeWeight, SRef)>>(); // TODO can use this approxiamtion for the number of edges

        //add nodes and edges to graph
        let node_mapping: HashMap<SRef, NodeIndex> = spec.all_streams().map(|sr| (sr, graph.add_node(sr))).collect();
        edges.iter().for_each(|(src, w, tar)| {
            graph.add_edge(node_mapping[src], node_mapping[tar], w.clone());
        });

        // Check well-formedness = no closed-walk with total weight of zero or positive
        Self::check_well_formedness(&graph)?;

        // Describe dependencies in HashMaps
        let mut accesses: HashMap<SRef, Vec<SRef>> = spec.all_streams().map(|sr| (sr, Vec::new())).collect();
        let mut accessed_by: HashMap<SRef, Vec<SRef>> = spec.all_streams().map(|sr| (sr, Vec::new())).collect();
        let mut aggregates: HashMap<SRef, Vec<(SRef, WRef)>> = HashMap::new();
        let mut aggregated_by: HashMap<SRef, Vec<(SRef, WRef)>> = HashMap::new();
        edges.iter().for_each(|(src, w, tar)| {
            let cur_accesses = accesses.get_mut(src).unwrap();
            if !cur_accesses.contains(tar) {
                cur_accesses.push(*tar);
            }
            let cur_accessed_by = accessed_by.get_mut(tar).unwrap();
            if !cur_accessed_by.contains(src) {
                cur_accessed_by.push(*src);
            }
            if let EdgeWeight::Aggr(wref) = w {
                let cur_aggregates = &mut (*aggregates.entry(*src).or_insert_with(Vec::new));
                if cur_aggregates.contains(&(*tar, *wref)) {
                    cur_aggregates.push((*tar, *wref));
                }
                let cur_aggregates_by = &mut (*aggregated_by.entry(*tar).or_insert_with(Vec::new));
                if cur_aggregates_by.contains(&(*src, *wref)) {
                    cur_aggregates_by.push((*src, *wref));
                }
            }
        });
        Ok(Dependencies { accesses, accessed_by, aggregates, aggregated_by, graph })
    }

    fn check_well_formedness(graph: &Graph<SRef, EdgeWeight>) -> Result<()> {
        let graph = graph_without_negative_offset_edges(graph);
        let graph = graph_without_close_edges(&graph);
        // check if cyclic
        if is_cyclic_directed(&graph) {
            Err(DependencyErr::WellFormedNess)
        } else {
            Ok(())
        }
    }

    fn collect_edges<M>(spec: &Hir<M>, src: SRef, expr: &Expression) -> Vec<(SRef, StreamAccessKind, SRef)>
    where
        M: WithIrExpr + HirMode + 'static,
    {
        match &expr.kind {
            ExpressionKind::StreamAccess(target, stream_access_kind, args) => {
                let mut args = args.iter().map(|arg| Self::collect_edges(spec, src, arg)).flatten().collect::<Vec<(
                    SRef,
                    StreamAccessKind,
                    SRef,
                )>>(
                );
                args.push((src, *stream_access_kind, *target));
                args
            }
            ExpressionKind::LoadConstant(_) => Vec::new(),
            ExpressionKind::ArithLog(_op, args) => {
                args.iter().flat_map(|a| Self::collect_edges(spec, src, a).into_iter()).collect()
            }
            ExpressionKind::Tuple(content) => content.iter().flat_map(|a| Self::collect_edges(spec, src, a)).collect(),
            ExpressionKind::Function { args, .. } => {
                args.iter().flat_map(|a| Self::collect_edges(spec, src, a)).collect()
            }
            ExpressionKind::Ite { condition, consequence, alternative } => Self::collect_edges(spec, src, condition)
                .into_iter()
                .chain(Self::collect_edges(spec, src, consequence).into_iter())
                .chain(Self::collect_edges(spec, src, alternative).into_iter())
                .collect(),
            ExpressionKind::TupleAccess(content, _n) => Self::collect_edges(spec, src, content),
            ExpressionKind::Widen(inner, _) => Self::collect_edges(spec, src, inner),
            ExpressionKind::Default { expr, default } => Self::collect_edges(spec, src, expr)
                .into_iter()
                .chain(Self::collect_edges(spec, src, default).into_iter())
                .collect(),
            ExpressionKind::ParameterAccess(_, _) => Vec::new(), //check this
        }
    }

    fn stream_access_kind_to_edge_weight(w: StreamAccessKind) -> EdgeWeight {
        match w {
            StreamAccessKind::Sync => EdgeWeight::Offset(0),
            StreamAccessKind::Offset(o) => match o {
                Offset::PastDiscreteOffset(o) => EdgeWeight::Offset(-i32::try_from(o).unwrap()),
                Offset::FutureDiscreteOffset(o) => {
                    if o == 0 {
                        EdgeWeight::Offset(i32::try_from(o).unwrap())
                    } else {
                        todo!("implement dependency analysis for positive future offsets")
                    }
                }
                _ => todo!("implement dependency analysis for real-time offsets"),
            },
            StreamAccessKind::Hold => EdgeWeight::Hold,

            StreamAccessKind::SlidingWindow(wref) => EdgeWeight::Aggr(wref),
            StreamAccessKind::DiscreteWindow(wref) => EdgeWeight::Aggr(wref),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::hir::modes::IrExpression;
    use crate::hir::SRef;
    use crate::hir::WRef;
    use crate::parse::{parse, SourceMapper};
    use crate::reporting::Handler;
    use crate::FrontendConfig;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn check_graph_for_spec(
        spec: &str,
        accesses: Option<HashMap<SRef, Vec<SRef>>>,
        accessed_by: Option<HashMap<SRef, Vec<SRef>>>,
        aggregates: Option<HashMap<SRef, Vec<(SRef, WRef)>>>,
        aggregated_by: Option<HashMap<SRef, Vec<(SRef, WRef)>>>,
    ) {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<IrExpression>::transform_expressions(ast, &handler, &config);
        let deps = Dependencies::analyze(&hir);
        if let Ok(deps) = deps {
            deps.accesses.iter().for_each(|(sr, accesses_hir)| {
                let accesses_reference = accesses.as_ref().unwrap().get(sr).unwrap();
                assert_eq!(accesses_hir.len(), accesses_reference.len(), "sr: {}", sr);
                accesses_hir.iter().for_each(|sr| assert!(accesses_reference.contains(sr), "sr: {}", sr));
            });
            deps.accessed_by.iter().for_each(|(sr, accessed_by_hir)| {
                let accessed_by_reference = accessed_by.as_ref().unwrap().get(sr).unwrap();
                assert_eq!(accessed_by_hir.len(), accessed_by_reference.len(), "sr: {}", sr);
                accessed_by_hir.iter().for_each(|sr| assert!(accessed_by_reference.contains(sr), "sr: {}", sr));
            });
            deps.aggregates.iter().for_each(|(sr, aggregates_hir)| {
                let aggregates_reference = aggregates.as_ref().unwrap().get(sr).unwrap();
                assert_eq!(aggregates_hir.len(), aggregates_reference.len(), "test");
                aggregates_hir.iter().for_each(|lookup| assert!(aggregates_reference.contains(lookup)));
            });
            deps.aggregated_by.iter().for_each(|(sr, aggregated_by_hir)| {
                let aggregated_by_reference = aggregated_by.as_ref().unwrap().get(sr).unwrap();
                assert_eq!(aggregated_by_hir.len(), aggregated_by_reference.len(), "test");
                aggregated_by_hir.iter().for_each(|lookup| assert!(aggregated_by_reference.contains(lookup)));
            });
        } else {
            assert!(accesses == None && accessed_by == None && aggregates == None && aggregated_by == None)
        }
    }

    #[test]
    fn self_ref() {
        let spec = "input a: Int8\noutput b: Int8 := a+b";
        let accesses = None;
        let accessed_by = None;
        let aggregates = None;
        let aggregated_by = None;
        check_graph_for_spec(spec, accesses, accessed_by, aggregates, aggregated_by);
    }
    #[test]
    fn simple_cycle() {
        let spec = "input a: Int8\noutput b: Int8 := a+d\noutput c: Int8 := b\noutput d: Int8 := c";
        check_graph_for_spec(spec, None, None, None, None);
    }

    #[test]
    fn linear_should_be_no_problem() {
        let spec = "input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let accesses = Some(
            vec![
                (name_mapping["a"], vec![]),
                (name_mapping["b"], vec![name_mapping["a"]]),
                (name_mapping["c"], vec![name_mapping["b"]]),
                (name_mapping["d"], vec![name_mapping["c"]]),
            ]
            .into_iter()
            .collect(),
        );
        let accessed_by = Some(
            vec![
                (name_mapping["a"], vec![name_mapping["b"]]),
                (name_mapping["b"], vec![name_mapping["c"]]),
                (name_mapping["c"], vec![name_mapping["d"]]),
                (name_mapping["d"], vec![]),
            ]
            .into_iter()
            .collect(),
        );
        let aggregates = Some(
            vec![
                (name_mapping["a"], vec![]),
                (name_mapping["b"], vec![]),
                (name_mapping["c"], vec![]),
                (name_mapping["d"], vec![]),
            ]
            .into_iter()
            .collect(),
        );
        let aggregated_by = Some(
            vec![
                (name_mapping["a"], vec![]),
                (name_mapping["b"], vec![]),
                (name_mapping["c"], vec![]),
                (name_mapping["d"], vec![]),
            ]
            .into_iter()
            .collect(),
        );
        check_graph_for_spec(spec, accesses, accessed_by, aggregates, aggregated_by);
    }

    #[test]
    fn negative_cycle_should_be_no_problem() {
        let spec = "output a: Int8 := a.offset(by: -1).defaults(to: 0)";
        let name_mapping = vec![("a", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let accesses = Some(vec![(name_mapping["a"], vec![name_mapping["a"]])].into_iter().collect());
        let accessed_by = Some(vec![(name_mapping["a"], vec![name_mapping["a"]])].into_iter().collect());
        let aggregates = Some(vec![(name_mapping["a"], vec![])].into_iter().collect());
        let aggregated_by = Some(vec![(name_mapping["a"], vec![])].into_iter().collect());
        check_graph_for_spec(spec, accesses, accessed_by, aggregates, aggregated_by);
    }

    #[test]
    fn self_sliding_window() {
        let spec = "output a: Int8 := a.aggregate(over: 1s, using: sum)";
        check_graph_for_spec(spec, None, None, None, None);
    }

    // #[test]
    // #[ignore] // Graph Analysis cannot handle positive edges; not required for this branch.
    // fn positive_cycle_should_cause_a_warning() {
    //     chek_graph_for_spec("output a: Int8 := a[1]", false);
    // }

    #[test]
    fn self_loop() {
        let spec = "input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c\noutput e: Int8 := e";
        check_graph_for_spec(spec, None, None, None, None)
    }

    #[test]
    fn parallel_edges_in_a_cycle() {
        let spec = "input a: Int8\noutput b: Int8 := a+d+d\noutput c: Int8 := b\noutput d: Int8 := c";
        check_graph_for_spec(spec, None, None, None, None);
    }

    #[test]
    #[ignore]
    fn spawn_self_ref() {
        let spec = "input a: Int8\noutput b(para) {spawn a if b(para) > 6} := a + para";
        check_graph_for_spec(spec, None, None, None, None);
    }
    #[test]
    #[ignore]
    fn filter_self_ref() {
        let spec = "input a: Int8\noutput b {filter b > 6} := a + 5";
        check_graph_for_spec(spec, None, None, None, None);
    }

    #[test]
    #[ignore]
    fn close_self_ref() {
        let spec = "input a: Int8\noutput b {close b > 6} := a + 5";
        let name_mapping =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let accesses = Some(
            vec![(name_mapping["a"], vec![]), (name_mapping["b"], vec![name_mapping["a"], name_mapping["b"]])]
                .into_iter()
                .collect(),
        );
        let accessed_by = Some(
            vec![(name_mapping["a"], vec![]), (name_mapping["a"], vec![name_mapping["a"], name_mapping["b"]])]
                .into_iter()
                .collect(),
        );
        let aggregates = Some(vec![(name_mapping["a"], vec![]), (name_mapping["b"], vec![])].into_iter().collect());
        let aggregated_by = Some(vec![(name_mapping["a"], vec![]), (name_mapping["b"], vec![])].into_iter().collect());
        check_graph_for_spec(spec, accesses, accessed_by, aggregates, aggregated_by);
    }
}
