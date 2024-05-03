use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::io::BufWriter;

use dot::{LabelText, Style};
use itertools::Itertools;
use serde::{Serialize, Serializer};
use serde_json::{json, to_string_pretty};

use super::{
    ActivationCondition, Mir, Origin, PacingType, StreamAccessKind, StreamReference, TriggerReference, WindowReference,
};

/// Represents the dependency graph of the specification
#[derive(Debug, Clone)]
pub struct DependencyGraph<'a> {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    infos: HashMap<Node, NodeInformation<'a>>,
}

impl<'a> DependencyGraph<'a> {
    pub(super) fn new(mir: &'a Mir) -> Self {
        let stream_nodes = mir
            .inputs
            .iter()
            .map(|i| i.reference)
            .chain(mir.outputs.iter().filter(|o| !o.is_trigger()).map(|o| o.reference))
            .map(Node::Stream);

        let window_nodes = mir.sliding_windows.iter().map(|w| Node::Window(w.reference));

        let trigger_nodes = mir
            .triggers
            .iter()
            .map(|trigger| Node::Trigger(trigger.trigger_reference));

        let nodes: Vec<_> = stream_nodes.chain(window_nodes).chain(trigger_nodes).collect();

        let edges = edges(mir);

        let infos = nodes.iter().map(|node| (*node, node_infos(mir, *node))).collect();

        Self { nodes, edges, infos }
    }

    /// Returns the dependency graph in the graphviz dot-format
    pub fn dot(&self) -> String {
        let res = Vec::new();
        let mut res_writer = BufWriter::new(res);
        dot::render(self, &mut res_writer).unwrap();
        String::from_utf8(res_writer.into_inner().unwrap()).unwrap()
    }

    /// Returns the dependency graph in a json-format
    pub fn json(&self) -> String {
        let infos = self
            .infos
            .iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect::<HashMap<_, _>>();

        let json_value = json!({
            "edges": self.edges,
            "nodes": infos
        });

        to_string_pretty(&json_value).unwrap()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Node {
    Stream(StreamReference),
    Trigger(TriggerReference),
    Window(WindowReference),
}

impl From<StreamReference> for Node {
    fn from(s: StreamReference) -> Self {
        Node::Stream(s)
    }
}

impl From<WindowReference> for Node {
    fn from(w: WindowReference) -> Self {
        Node::Window(w)
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Stream(StreamReference::In(i)) => write!(f, "In_{i}"),
            Node::Stream(StreamReference::Out(i)) => write!(f, "Out_{i}"),
            Node::Window(WindowReference::Sliding(i)) => write!(f, "SW_{i}"),
            Node::Window(WindowReference::Discrete(i)) => write!(f, "DW_{i}"),
            Node::Window(WindowReference::Instance(i)) => write!(f, "IA_{i}"),
            Node::Trigger(i) => write!(f, "T_{i}"),
        }
    }
}

impl Serialize for Node {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct Edge {
    from: Node,
    with: EdgeType,
    to: Node,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "type")]
enum EdgeType {
    Access { kind: StreamAccessKind, origin: Origin },
    Spawn,
    Eval,
}

impl Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EdgeType::Access {
                kind: StreamAccessKind::Sync,
                ..
            } => "Sync".into(),
            EdgeType::Access {
                kind: StreamAccessKind::Hold,
                ..
            } => "Hold".into(),
            EdgeType::Access {
                kind: StreamAccessKind::Offset(o),
                ..
            } => format!("Offset({o})"),
            EdgeType::Spawn => "Spawn".into(),
            EdgeType::Eval => "Eval".into(),
            EdgeType::Access {
                kind: StreamAccessKind::InstanceAggregation(_),
                ..
            } => "Instances".into(),
            // no label on window access edges
            EdgeType::Access {
                kind: StreamAccessKind::DiscreteWindow(_),
                ..
            }
            | EdgeType::Access {
                kind: StreamAccessKind::SlidingWindow(_),
                ..
            } => "".into(),
            EdgeType::Access {
                kind: StreamAccessKind::Get,
                ..
            } => "Get".into(),
            EdgeType::Access {
                kind: StreamAccessKind::Fresh,
                ..
            } => "Fresh".into(),
        };

        write!(f, "{s}")
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
enum NodeInformation<'a> {
    Input {
        reference: StreamReference,
        stream_name: &'a str,
        memory_bound: u32,
        value_ty: String,
    },

    Output {
        reference: StreamReference,
        stream_name: &'a str,
        eval_layer: usize,
        memory_bound: u32,
        pacing_ty: String,
        spawn_ty: String,
        value_ty: String,
    },

    Window {
        reference: WindowReference,
        operation: String,
        duration: String,
        pacing_ty: String,
        memory_bound: u32,
    },
}

fn node_infos(mir: &Mir, node: Node) -> NodeInformation {
    match node {
        Node::Stream(sref) => stream_infos(mir, sref),
        Node::Window(wref) => window_infos(mir, wref),
        Node::Trigger(sref) => stream_infos(mir, mir.triggers[sref].output_reference),
    }
}

fn stream_infos(mir: &Mir, sref: StreamReference) -> NodeInformation {
    let stream = mir.stream(sref);

    let stream_name = stream.name();
    let eval_layer: usize = stream.eval_layer().into();
    let memory_bound = stream.values_to_memorize().unwrap();
    let value_ty = stream.ty();
    let value_str = value_ty.to_string();

    match sref {
        StreamReference::In(_) => NodeInformation::Input {
            reference: sref,
            stream_name,
            memory_bound,
            value_ty: value_str,
        },
        StreamReference::Out(_) => {
            let output = mir.output(sref);
            let pacing_str = mir.display(&output.eval.eval_pacing).to_string();
            let spawn_str = mir.display(&output.spawn.pacing).to_string();

            NodeInformation::Output {
                reference: sref,
                stream_name,
                eval_layer,
                memory_bound,
                pacing_ty: pacing_str,
                spawn_ty: spawn_str,
                value_ty: value_str,
            }
        },
    }
}

fn window_infos(mir: &Mir, wref: WindowReference) -> NodeInformation {
    let window = mir.window(wref);
    let operation_str = window.op().to_string();
    let duration_str = match wref {
        WindowReference::Sliding(_) => {
            let duration = mir.sliding_window(wref).duration;
            format!("{}s", duration.as_secs_f64())
        },
        WindowReference::Discrete(_) => {
            let duration = mir.discrete_window(wref).duration;
            format!("{duration} values")
        },

        WindowReference::Instance(_) => {
            let selection = mir.instance_aggregation(wref).selection;
            format!("{selection} instances")
        },
    };
    let caller = mir.output(window.caller());

    let origin = caller
        .accesses
        .iter()
        .flat_map(|(_, accesses)| accesses)
        .find(|(_, kind)| {
            *kind == StreamAccessKind::SlidingWindow(wref) || *kind == StreamAccessKind::DiscreteWindow(wref)
        })
        .expect("access has to exist")
        .0;

    let pacing = match origin {
        Origin::Spawn => &caller.spawn.pacing,
        Origin::Filter(_) | Origin::Eval(_) => &caller.eval.eval_pacing,
        Origin::Close => &caller.close.pacing,
    };

    let pacing_str = mir.display(pacing).to_string();
    let memory_bound = window.memory_bound().unwrap();

    NodeInformation::Window {
        reference: wref,
        operation: operation_str,
        duration: duration_str,
        pacing_ty: pacing_str,
        memory_bound,
    }
}

fn edges(mir: &Mir) -> Vec<Edge> {
    let input_accesses = mir.inputs.iter().map(|input| (input.reference, &input.accessed_by));
    let output_accesses = mir.outputs.iter().map(|output| (output.reference, &output.accessed_by));
    let all_accesses = input_accesses.chain(output_accesses);
    let out_to_trig: &HashMap<_, _> = &(mir
        .triggers
        .iter()
        .map(|t| (t.output_reference, t.trigger_reference))
        .collect());

    let access_edges = all_accesses.flat_map(|(source_ref, accesses)| {
        let source = out_to_trig
            .get(&source_ref)
            .map(|t| Node::Trigger(*t))
            .unwrap_or_else(|| Node::Stream(source_ref));
        accesses.iter().flat_map(move |(target_ref, access_kinds)| {
            let target = out_to_trig
                .get(target_ref)
                .map(|t| Node::Trigger(*t))
                .unwrap_or_else(|| Node::Stream(*target_ref));
            access_kinds.iter().flat_map(move |&(origin, kind)| match kind {
                StreamAccessKind::SlidingWindow(w) | StreamAccessKind::DiscreteWindow(w) => {
                    let window = mir.window(w);
                    let with = EdgeType::Access { origin, kind };
                    vec![
                        Edge {
                            from: Node::Stream(window.caller()),
                            with: with.clone(),
                            to: Node::Window(w),
                        },
                        Edge {
                            from: Node::Window(w),
                            with,
                            to: Node::Stream(window.target()),
                        },
                    ]
                },
                StreamAccessKind::Fresh
                | StreamAccessKind::Get
                | StreamAccessKind::Hold
                | StreamAccessKind::Offset(_)
                | StreamAccessKind::InstanceAggregation(_)
                | StreamAccessKind::Sync => {
                    vec![Edge {
                        from: target,
                        with: EdgeType::Access { origin, kind },
                        to: source,
                    }]
                },
            })
        })
    });

    let spawn_edges = mir.outputs.iter().flat_map(|output| {
        let source = out_to_trig
            .get(&output.reference)
            .map(|t| Node::Trigger(*t))
            .unwrap_or_else(|| Node::Stream(output.reference));
        match &output.spawn.pacing {
            PacingType::Event(ac) => flatten_ac(ac)
                .into_iter()
                .map(|input| Edge {
                    from: source,
                    with: EdgeType::Spawn,
                    to: Node::Stream(input),
                })
                .collect(),
            PacingType::LocalPeriodic(_) | PacingType::GlobalPeriodic(_) | PacingType::Constant => vec![],
        }
    });

    let ac_edges = mir.outputs.iter().flat_map(|output| {
        let source = out_to_trig
            .get(&output.reference)
            .map(|t| Node::Trigger(*t))
            .unwrap_or_else(|| Node::Stream(output.reference));
        match &output.eval.eval_pacing {
            PacingType::Event(ac) => flatten_ac(ac)
                .into_iter()
                .map(|input| Edge {
                    from: source,
                    with: EdgeType::Eval,
                    to: Node::Stream(input),
                })
                .collect(),
            PacingType::LocalPeriodic(_) | PacingType::GlobalPeriodic(_) | PacingType::Constant => vec![],
        }
    });

    access_edges.chain(spawn_edges).chain(ac_edges).collect()
}

fn inner_flatten_ac(ac: &ActivationCondition) -> Vec<StreamReference> {
    match ac {
        ActivationCondition::Disjunction(xs) | ActivationCondition::Conjunction(xs) => {
            xs.iter().flat_map(flatten_ac).collect()
        },
        ActivationCondition::Stream(s) => vec![*s],
        ActivationCondition::True => vec![],
    }
}

fn flatten_ac(ac: &ActivationCondition) -> Vec<StreamReference> {
    let mut vec = inner_flatten_ac(ac);
    vec.sort();
    vec.dedup();
    vec
}

impl<'a> dot::Labeller<'a, Node, Edge> for DependencyGraph<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("dependency_graph").unwrap()
    }

    fn node_id(&'a self, n: &Node) -> dot::Id<'a> {
        let id = n.to_string();
        dot::Id::new(id).unwrap()
    }

    fn node_label<'b>(&'b self, n: &Node) -> LabelText<'b> {
        let infos = self.infos.get(n).unwrap();

        let label_text = match infos {
            NodeInformation::Input {
                stream_name,
                memory_bound,
                value_ty,
                reference: _,
            } => {
                format!("{stream_name}: {value_ty}<br/>Memory Bound: {memory_bound}")
            },
            NodeInformation::Output {
                stream_name,
                eval_layer,
                memory_bound,
                pacing_ty,
                spawn_ty,
                value_ty,
                reference: _,
            } => {
                format!(
                    "{stream_name}: {value_ty}<br/>\
Pacing: {pacing_ty}<br/>\
Spawn: {spawn_ty}<br/>\
Memory Bound: {memory_bound}<br/>\
Layer {eval_layer}"
                )
            },
            NodeInformation::Window {
                reference,
                operation,
                duration,
                pacing_ty: _,
                memory_bound: _,
            } => format!("Window {reference}<br/>Window Operation: {operation}<br/>Duration: {duration}"),
        };

        dot::LabelText::HtmlStr(label_text.into())
    }

    fn edge_label<'b>(&'b self, edge: &Edge) -> LabelText<'b> {
        dot::LabelText::LabelStr(edge.with.to_string().into())
    }

    fn edge_style(&self, edge: &Edge) -> Style {
        match &edge.with {
            EdgeType::Access { kind, origin: _ } => match kind {
                StreamAccessKind::Get | StreamAccessKind::Fresh | StreamAccessKind::Hold => Style::Dashed,
                StreamAccessKind::Sync
                | StreamAccessKind::InstanceAggregation(_)
                | StreamAccessKind::Offset(_)
                | StreamAccessKind::DiscreteWindow(_)
                | StreamAccessKind::SlidingWindow(_) => Style::None,
            },
            EdgeType::Spawn | EdgeType::Eval => Style::Dotted,
        }
    }

    fn node_shape(&self, node: &Node) -> Option<LabelText<'_>> {
        let shape_str = match node {
            Node::Stream(StreamReference::In(_)) => "box",
            Node::Stream(StreamReference::Out(_)) => "ellipse",
            Node::Trigger(_) => "octagon",
            Node::Window(_) => "note",
        };

        Some(dot::LabelText::LabelStr(shape_str.into()))
    }

    fn edge_end_arrow(&'a self, _e: &Edge) -> dot::Arrow {
        dot::Arrow::none()
    }

    fn edge_start_arrow(&'a self, _e: &Edge) -> dot::Arrow {
        dot::Arrow::normal()
    }
}

impl<'a> dot::GraphWalk<'a, Node, Edge> for DependencyGraph<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Node> {
        Cow::Borrowed(&self.nodes)
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge> {
        // all the sync and offset edges
        let ac_accesses = self
            .edges
            .iter()
            .filter(|edge| {
                matches!(
                    edge.with,
                    EdgeType::Access {
                        kind: StreamAccessKind::Sync,
                        ..
                    } | EdgeType::Access {
                        kind: StreamAccessKind::Offset(_),
                        ..
                    }
                )
            })
            .map(|edge| (&edge.from, &edge.to))
            .collect::<HashSet<_>>();

        let edges = self
            .edges
            .iter()
            // remove edges that have the same access kind but different origins, because
            // the origin is not displayed in the dot-representation
            .unique_by(|edge| {
                (
                    edge.from,
                    edge.to,
                    match edge.with {
                        EdgeType::Access { kind, origin: _ } => Some(kind),
                        EdgeType::Spawn | EdgeType::Eval => None,
                    },
                )
            })
            // in the dot format, we only want to render eval edges, if the edge it not already covered by sync or offset edges
            .filter(|edge| {
                match edge.with {
                    EdgeType::Access{..} | EdgeType::Spawn => true,
                    EdgeType::Eval => !ac_accesses.contains(&(&edge.from, &edge.to)),
                }
            })
            .cloned()
            .collect();
        Cow::Owned(edges)
    }

    fn source(&self, e: &Edge) -> Node {
        // because we add the arrows the wrong way round (see edge style)
        e.to
    }

    fn target(&self, e: &Edge) -> Node {
        // because we add the arrows the wrong way round (see edge style)
        e.from
    }
}

#[cfg(test)]
mod tests {
    use rtlola_parser::ParserConfig;

    use super::*;
    use crate::parse;

    macro_rules! build_node {
        ( In($i:expr) ) => {
            Node::Stream(StreamReference::In($i))
        };
        ( Out($i:expr) ) => {
            Node::Stream(StreamReference::Out($i))
        };
        ( T($i:expr) ) => {
            Node::Trigger($i)
        };
        ( SW($i:expr) ) => {
            Node::Window(WindowReference::Sliding($i))
        };
        ( DW($i:expr) ) => {
            Node::Window(WindowReference::Discrete($i))
        };
    }

    macro_rules! build_edge_kind {
        ( Spawn ) => {
            EdgeType::Spawn
        };
        ( Eval ) => {
            EdgeType::Eval
        };
        ( SW, $i:expr, $origin:ident $(, $origin_i:expr )? ) => {
            EdgeType::Access{origin: Origin::$origin$(($origin_i))?, kind: StreamAccessKind::SlidingWindow(WindowReference::Sliding($i))}
        };
        ( DW, $i:expr, $origin:ident ) => {
            EdgeType::Access{origin: Origin::&origin, kind: StreamAccessKind::DiscreteWindow(WindowReference::Discrete($i))}
        };
        ( $sak:ident, $origin:ident $(, $origin_i:expr )? ) => {
            EdgeType::Access{origin: Origin::$origin$(($origin_i))?, kind: StreamAccessKind::$sak}
        };
    }

    // https://stackoverflow.com/a/34324856
    macro_rules! count {
        () => (0usize);
        ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
    }

    macro_rules! test_dependency_graph {
        ( $name:ident, $spec:literal, $( $edge_from_ty:ident($edge_from_i:expr)$(:$origin:ident$(($origin_i:expr))?)? => $edge_to_ty:ident($edge_to_i:expr) : $with:ident $(($p:expr))? , )+ ) => {

            #[test]
            fn $name() {
                let config = ParserConfig::for_string($spec.into());
                let mir = parse(&config).expect("should parse");
                let dep_graph = mir.dependency_graph();
                let edges = &dep_graph.edges;
                $(
                    let from_node = build_node!($edge_from_ty($edge_from_i));
                    let to_node = build_node!($edge_to_ty($edge_to_i));
                    let with = build_edge_kind!($with $(,$p)? $(,$origin $(,$origin_i)?)?);
                    let expected_edge = Edge {
                        from: from_node, to: to_node, with
                    };
                    assert!(edges.iter().any(|edge| *edge == expected_edge), "specification did not contain expected edge {:#?}", expected_edge);
                )+
                assert!(edges.len() == count!($($with)+), "dependency graph had unwanted additional edges");
            }
        };
    }

    test_dependency_graph!(simple,
        "input a : UInt64
        input b : UInt64
        output c := a + b",
        Out(0):Eval(0) => In(0) : Sync,
        Out(0):Eval(0) => In(1) : Sync,
        Out(0) => In(0) : Eval,
        Out(0) => In(1) : Eval,
    );

    test_dependency_graph!(trigger,
        "input a : UInt64
        trigger a > 5",
        T(0):Filter(0) => In(0) : Sync,
        T(0) => In(0) : Eval,
    );

    test_dependency_graph!(more_complex,
        "input a : UInt64
        input b : UInt64
        output c := a + b.hold().defaults(to:0)
        output d@1Hz := a.aggregate(over:5s, using:count)
        trigger d < 5",
        Out(0):Eval(0) => In(0) : Sync,
        Out(0):Eval(0) => In(1) : Hold,
        Out(1):Eval(0) => SW(0) : SW(0),
        SW(0):Eval(0) => In(0) : SW(0),
        T(0):Filter(0) => Out(1) : Sync,
        Out(0) => In(0) : Eval,
    );

    test_dependency_graph!(ac,
        "input a : UInt64
        input b : UInt64
        output c @(a||b) := 0
        output d @(a&&b) := a
        ",
        Out(1):Eval(0) => In(0) : Sync,
        Out(0) => In(0) : Eval,
        Out(0) => In(1) : Eval,
        Out(1) => In(0) : Eval,
        Out(1) => In(1) : Eval,
    );

    test_dependency_graph!(spawn,
        "input a : UInt64
        input b : UInt64
        output c(x)
            spawn with a
            eval with b when x == a
        ",
        Out(0) => In(0) : Spawn,
        Out(0) => In(0) : Eval,
        Out(0) => In(1) : Eval,
        Out(0):Filter(0) => In(0) : Sync,
        Out(0):Eval(0) => In(1) : Sync,
        Out(0):Spawn => In(0) : Sync,
    );

    test_dependency_graph!(multiple_evals,
        "input a : UInt64
        input b : UInt64
        output c
            eval @(a&&b) when a == 0 with 0
            eval @(a&&b) when b == 0 with 1
            eval @(a&&b) when a + b == 1 with a  
        ",
        Out(0) => In(0) : Eval,
        Out(0) => In(1) : Eval,
        Out(0):Filter(0) => In(0) : Sync,
        Out(0):Filter(1) => In(1) : Sync,
        Out(0):Filter(2) => In(0) : Sync,
        Out(0):Filter(2) => In(1) : Sync,
        Out(0):Eval(2) => In(0) : Sync,
    );
}
