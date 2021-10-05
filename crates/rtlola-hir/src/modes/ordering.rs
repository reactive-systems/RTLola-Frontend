use std::collections::HashMap;

use petgraph::algo::is_cyclic_directed;
use petgraph::Outgoing;
use serde::{Deserialize, Serialize};

use super::{Ordered, OrderedTrait, TypedTrait};
use crate::hir::{Hir, SRef};
use crate::modes::dependencies::ExtendedDepGraph;
use crate::modes::{DepAnaTrait, DependencyGraph, HirMode};

impl OrderedTrait for Ordered {
    fn stream_layers(&self, sr: SRef) -> StreamLayers {
        match self.event_layers.get(&sr) {
            Some(layer) => *layer,
            None => self.periodic_layers[&sr],
        }
    }
}

/// Represents a layer indicating the position when an expression can be evaluated
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer(usize);

/// Wrapper to collect the layer when a stream instance is spawned and evaluated
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamLayers {
    spawn: Layer,
    evaluation: Layer,
}

impl StreamLayers {
    /// Produces the wrapper [StreamLayers] for a given spawn and evaluation layer
    pub(crate) fn new(spawn_layer: Layer, evaluation_layer: Layer) -> StreamLayers {
        StreamLayers {
            spawn: spawn_layer,
            evaluation: evaluation_layer,
        }
    }

    /// Returns the layer when a stream is spawned
    pub fn spawn_layer(&self) -> Layer {
        self.spawn
    }

    /// Returns the layer when a new stream value is produced
    pub fn evaluation_layer(&self) -> Layer {
        self.evaluation
    }
}

impl From<Layer> for usize {
    fn from(layer: Layer) -> usize {
        layer.0
    }
}

impl Layer {
    /// Produces a [Layer]
    pub fn new(layer: usize) -> Self {
        Layer(layer)
    }

    /// Returns the [Layer] as `usize`
    pub fn inner(self) -> usize {
        self.0
    }
}

impl Ordered {
    /// Returns the spawn and evaluation layer of each stream
    ///
    /// This function analyzes the `spec` and returns the spawn and evaluation layer of each stream.
    /// The analysis splits the dependency graph into two subgraphs: one with the event-based streams and one with the periodic streams.
    /// From these two graphs, it computed the spawn and evaluation layer for event-based and periodic streams separately.
    pub(crate) fn analyze<M>(spec: &Hir<M>) -> Ordered
    where
        M: HirMode + DepAnaTrait + TypedTrait,
    {
        // split graph in periodic and event-based
        let (event_graph, periodic_graph) = spec.graph().split_graph(spec);
        let event_layers = Self::compute_layers(spec, &event_graph, true);
        let periodic_layers = Self::compute_layers(spec, &periodic_graph, false);
        Ordered {
            event_layers,
            periodic_layers,
        }
    }

    /// Returns the spawn and evaluation layer of for either each event-based stream or periodic stream
    ///
    /// This function splits the dependency graph into two subgraphs and returns the spawn and evaluation layer of each stream.
    /// The first graph only contains the lookups occurring in the spawn part of the stream template.
    /// The analysis computes from this graph the spawn layers of each stream.
    /// The function computes from the second graph the evaluation layers of each stream.
    fn compute_layers<M>(spec: &Hir<M>, graph: &DependencyGraph, is_event: bool) -> HashMap<SRef, StreamLayers>
    where
        M: HirMode + DepAnaTrait + TypedTrait,
    {
        // Prepare graphs
        let graph = graph.without_negative_offset_edges().without_close_edges();
        let spawn_graph = graph.only_spawn_edges();
        debug_assert!(
            !is_cyclic_directed(&graph),
            "This should be already checked in the dependency analysis."
        );

        // start analysis
        let mut evaluation_layers = if is_event {
            spec.inputs()
                .map(|i| (i.sr, Layer::new(0)))
                .collect::<HashMap<SRef, Layer>>()
        } else {
            HashMap::new()
        };
        let mut spawn_layers = if is_event {
            spec.inputs()
                .map(|i| (i.sr, Layer::new(0)))
                .collect::<HashMap<SRef, Layer>>()
        } else {
            HashMap::new()
        };
        while graph.node_count() != evaluation_layers.len() {
            // build spawn layers
            spawn_graph.node_indices().for_each(|node| {
                let sref = spawn_graph.node_weight(node).unwrap();
                if !spawn_layers.contains_key(sref) {
                    let neighbor_layers: Vec<_> = spawn_graph
                        .neighbors_directed(node, Outgoing)
                        .map(|outgoing_neighbor| {
                            evaluation_layers
                                .get(spawn_graph.node_weight(outgoing_neighbor).unwrap())
                                .copied()
                        })
                        .collect();
                    let computed_spawn_layer = if neighbor_layers.is_empty() {
                        Some(Layer::new(0))
                    } else {
                        neighbor_layers
                            .into_iter()
                            .fold(Some(Layer::new(0)), |cur_res_layer, neighbor_layer| {
                                match (cur_res_layer, neighbor_layer) {
                                    (Some(cur_res_layer), Some(neighbor_layer)) => {
                                        Some(std::cmp::max(cur_res_layer, neighbor_layer))
                                    },
                                    _ => None,
                                }
                            })
                            .map(|layer| Layer::new(layer.inner() + 1))
                    };
                    if let Some(layer) = computed_spawn_layer {
                        spawn_layers.insert(*sref, layer);
                    }
                }
            });
            graph.node_indices().for_each(|node| {
                let sref = graph.node_weight(node).unwrap();
                // build evaluation layers
                if !evaluation_layers.contains_key(sref) && spawn_layers.contains_key(sref) {
                    // Layer for current stream check incoming
                    let neighbor_layers: Vec<_> = graph
                        .neighbors_directed(node, Outgoing)
                        .flat_map(|outgoing_neighbor| {
                            if outgoing_neighbor == node {
                                None
                            } else {
                                Some(outgoing_neighbor)
                            }
                        }) // delete self references
                        .map(|outgoing_neighbor| {
                            evaluation_layers
                                .get(graph.node_weight(outgoing_neighbor).unwrap())
                                .copied()
                        })
                        .collect();
                    let computed_evaluation_layer = if neighbor_layers.is_empty() {
                        if is_event {
                            Some(Layer::new(1))
                        } else {
                            Some(Layer::new(0))
                        }
                    } else {
                        neighbor_layers
                            .into_iter()
                            .fold(Some(Layer::new(0)), |cur_res_layer, neighbor_layer| {
                                match (cur_res_layer, neighbor_layer) {
                                    (Some(cur_res_layer), Some(neighbor_layer)) => {
                                        Some(std::cmp::max(cur_res_layer, neighbor_layer))
                                    },
                                    _ => None,
                                }
                            })
                            .map(|layer| Layer::new(layer.inner() + 1))
                    };
                    if let Some(layer) = computed_evaluation_layer {
                        let layer = if spawn_layers[sref] < layer {
                            layer
                        } else {
                            Layer::new(spawn_layers[sref].inner() + 1)
                        };
                        evaluation_layers.insert(*sref, layer);
                    }
                }
            });
        }
        evaluation_layers
            .into_iter()
            .map(|(key, evaluation_layer)| (key, StreamLayers::new(spawn_layers[&key], evaluation_layer)))
            .collect::<HashMap<SRef, StreamLayers>>()
    }
}

#[cfg(test)]
mod tests {
    use rtlola_parser::{parse, ParserConfig};

    use super::*;
    use crate::modes::BaseMode;
    fn check_eval_order_for_spec(
        spec: &str,
        ref_event_layers: HashMap<SRef, StreamLayers>,
        ref_periodic_layers: HashMap<SRef, StreamLayers>,
    ) {
        let ast = parse(ParserConfig::for_string(spec.to_string())).unwrap_or_else(|e| panic!("{:?}", e));
        let hir = Hir::<BaseMode>::from_ast(ast)
            .unwrap()
            .analyze_dependencies()
            .unwrap()
            .check_types()
            .unwrap();
        let order = Ordered::analyze(&hir);
        let Ordered {
            event_layers,
            periodic_layers,
        } = order;
        assert_eq!(event_layers.len(), ref_event_layers.len());
        event_layers.iter().for_each(|(sr, layers)| {
            let ref_layers = &ref_event_layers[sr];
            assert_eq!(ref_layers.spawn_layer(), layers.spawn_layer());
            assert_eq!(ref_layers.evaluation_layer(), layers.evaluation_layer());
        });
        assert_eq!(periodic_layers.len(), ref_periodic_layers.len());
        periodic_layers.iter().for_each(|(sr, layers)| {
            let ref_layers = &ref_periodic_layers[sr];
            assert_eq!(ref_layers.spawn_layer(), layers.spawn_layer());
            assert_eq!(ref_layers.evaluation_layer(), layers.evaluation_layer());
        });
    }

    #[test]
    fn synchronous_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a\noutput c:UInt8 := b";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0)), ("c", SRef::Out(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn hold_lookup() {
        let spec =
            "input a: UInt8\ninput b:UInt8\noutput c: UInt8 := a.hold().defaults(to: 0) + b\noutput d: UInt8 := c.hold().defaults(to: 0) + a";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn offset_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by: -1).defaults(to: 0)\noutput c: UInt8 := b.offset(by: -1).defaults(to: 0)";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0)), ("c", SRef::Out(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn sliding_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over: 1s, using: sum)\noutput c: UInt8 := a + 3\noutput d: UInt8 @1Hz := c.aggregate(over: 1s, using: sum)\noutput e: UInt8 := b + d\noutput f: UInt8 @2Hz := e.hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
            ("e", SRef::Out(3)),
            ("f", SRef::Out(4)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["f"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn discrete_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over_discrete: 5, using: sum)\noutput c: UInt8 := a + 3\noutput d: UInt8 @1Hz := c.aggregate(over_discrete: 5, using: sum)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
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
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
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
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn lookup_chain() {
        let spec = "input a: Int8\noutput b: Int8 := a + d.hold().defaults(to:0)\noutput c: Int8 := b\noutput d: Int8 := c.offset(by:-4).defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(3))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn multiple_input_stream() {
        let spec = "input a: Int8\ninput b: Int8\noutput c: Int8 := a + b.hold().defaults(to:0)\noutput d: Int8 := a + c.offset(by: -1).defaults(to: 0)\noutput e: Int8 := c + 3\noutput f: Int8 := c + 6\noutput g: Int8 := b + 3\noutput h: Int8 := g + f";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
            ("f", SRef::Out(3)),
            ("g", SRef::Out(4)),
            ("h", SRef::Out(5)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["f"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["g"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["h"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn event_and_periodic_stream_mix() {
        let spec =
            "input a : Int8 \ninput b :Int8\noutput c @2Hz := a.hold().defaults(to: 0) + 3\noutput d @1Hz := a.hold().defaults(to: 0) + c\noutput e := a + b";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn negative_and_postive_lookups_as_loop() {
        let spec = "input a: Int8\noutput b: Int8 := a + d.offset(by:-1).defaults(to:0)\noutput c: Int8 := b\noutput d: Int8 := c";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
    #[test]
    fn sliding_windows_chain_and_hold_lookup() {
        let spec = "input a: Int8\noutput b@1Hz := a.aggregate(over: 1s, using: sum) + d.offset(by: -1).defaults(to: 0)\noutput c@2Hz := b.aggregate(over: 1s, using: sum)\noutput d@2Hz := b.hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![(sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0)))]
            .into_iter()
            .collect();
        let periodic_layers = vec![
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn simple_chain_with_parameter() {
        let spec = "input a: Int8\noutput b := a + 5\noutput c(para) spawn with b := para + a";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0)), ("c", SRef::Out(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(2), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn lookup_chain_with_parametrization() {
        let spec = "input a: Int8\noutput b(para) spawn with a if a > 6 := a + para\noutput c(para) spawn with a if a > 6 := a + b(para)\noutput d(para) spawn with a if a > 6 := a + c(para)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(1), Layer::new(2))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(3))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(1), Layer::new(4))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn parameter_loop_with_lookup_in_close() {
        let spec = "input a: Int8\ninput b: Int8\noutput c(p) spawn with a if a < b := p + b + g(p).hold().defaults(to: 0)\noutput d(p) spawn with b if c(4).hold().defaults(to: 0) < 4 := b + 5\noutput e(p) spawn with b := d(p).hold().defaults(to: 0) + b\noutput f(p) spawn with b filter e(p).hold().defaults(to: 0) < 6 := b + 5\noutput g(p) spawn with b close f(p).hold().defaults(to: 0) < 6 := b + 5";
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
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(3))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(4), Layer::new(5))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(1), Layer::new(6))),
            (sname_to_sref["f"], StreamLayers::new(Layer::new(1), Layer::new(7))),
            (sname_to_sref["g"], StreamLayers::new(Layer::new(1), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn parameter_nested_lookup_implicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(c(b).hold().defaults(to: 0)).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn parameter_nested_lookup_explicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(3))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(4))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn test_spawn_eventbased() {
        let spec = "input a: Int32\n\
                  output b(x: Int32) spawn with a := x + a";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(1), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn test_delay() {
        let spec = "input a: UInt64\n\
                            output a_counter: UInt64 @a := a_counter.offset(by: -1).defaults(to: 0) + 1\n\
                            output b(p: UInt64) @1Hz spawn with a_counter if a = 1 close if true then true else b(p) := a.hold(or: 0) == 2";
        let sname_to_sref = vec![("a", SRef::In(0)), ("a_counter", SRef::Out(0)), ("b", SRef::Out(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (
                sname_to_sref["a_counter"],
                StreamLayers::new(Layer::new(0), Layer::new(1)),
            ),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![(sname_to_sref["b"], StreamLayers::new(Layer::new(2), Layer::new(3)))]
            .into_iter()
            .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
}
