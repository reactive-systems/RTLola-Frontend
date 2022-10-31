use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;
use std::io::BufWriter;

use dot::{LabelText, Style};
use serde::{Serialize, Serializer};
use serde_json::{json, to_string_pretty};

use super::print::display_duration;
use super::{Mir, StreamAccessKind, StreamReference, TriggerReference, WindowReference};

/// Represents the dependency graph of the specification
#[derive(Debug, Clone)]
pub struct DependencyGraph<'a> {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    infos: HashMap<Node, NodeInformation<'a>>,
}

impl<'a> DependencyGraph<'a> {
    pub(super) fn new(mir: &'a Mir) -> Self {
        let stream_nodes = mir.all_streams().map(|s| Node::Stream(s));
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct EdgeType(StreamAccessKind);

impl From<StreamAccessKind> for EdgeType {
    fn from(value: StreamAccessKind) -> Self {
        EdgeType(value)
    }
}

impl Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self.0 {
            StreamAccessKind::Sync => "Sync".into(),
            StreamAccessKind::Hold => "Hold".into(),
            StreamAccessKind::Offset(o) => format!("Offset({o})"),
            StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => "".into(),
            StreamAccessKind::Get => todo!(),
            StreamAccessKind::Fresh => todo!(),
        };

        write!(f, "{s}")
    }
}

impl Serialize for EdgeType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
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
        value_ty: String,
        expression: String,
    },

    Trigger {
        idx: usize,
        eval_layer: usize,
        pacing_ty: String,
        value_ty: String,
        message: &'a str,
        expression: String,
    },

    Window {
        reference: WindowReference,
        operation: String,
        duration: String,
    },
}

fn node_infos(mir: &Mir, node: Node) -> NodeInformation {
    match node {
        Node::Stream(sref) => stream_infos(mir, sref),
        Node::Window(wref) => window_infos(mir, wref),
        Node::Trigger(tref) => trigger_infos(mir, tref),
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
        StreamReference::In(_) => {
            NodeInformation::Input {
                reference: sref,
                stream_name,
                memory_bound,
                value_ty: value_str,
            }
        },
        StreamReference::Out(_) => {
            let output = mir.output(sref);
            let pacing_str = mir.display(&output.eval.eval_pacing).to_string();
            let expr_str = mir.display(&output.eval.expression).to_string();

            NodeInformation::Output {
                reference: sref,
                stream_name,
                eval_layer,
                memory_bound,
                pacing_ty: pacing_str,
                value_ty: value_str,
                expression: expr_str,
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
            display_duration(duration)
        },
        WindowReference::Discrete(_) => {
            let duration = mir.discrete_window(wref).duration;
            format!("{duration} values")
        },
    };

    NodeInformation::Window {
        reference: wref,
        operation: operation_str,
        duration: duration_str,
    }
}

fn trigger_infos(mir: &Mir, tref: TriggerReference) -> NodeInformation {
    let trigger = &mir.triggers[tref];
    if let NodeInformation::Output {
        reference: _,
        stream_name: _,
        eval_layer,
        memory_bound: _,
        pacing_ty,
        value_ty,
        expression,
    } = stream_infos(mir, trigger.reference)
    {
        NodeInformation::Trigger {
            idx: tref,
            eval_layer,
            pacing_ty,
            value_ty,
            message: &trigger.message,
            expression,
        }
    } else {
        unreachable!("is NodeInformation::Stream");
    }
}

fn edges(mir: &Mir) -> Vec<Edge> {
    let input_accesses = mir.inputs.iter().map(|input| (input.reference, &input.accessed_by));
    let output_accesses = mir.outputs.iter().map(|output| (output.reference, &output.accessed_by));
    let all_accesses = input_accesses.chain(output_accesses);

    let access_edges = all_accesses.flat_map(|(source_ref, accesses)| {
        accesses.iter().flat_map(move |(target_ref, access_kinds)| {
            access_kinds.iter().filter_map(move |kind| {
                match kind {
                    // we remove edges for window accesses, because we add extra nodes for them
                    StreamAccessKind::SlidingWindow(_) | StreamAccessKind::DiscreteWindow(_) => None,
                    StreamAccessKind::Fresh
                    | StreamAccessKind::Get
                    | StreamAccessKind::Hold
                    | StreamAccessKind::Offset(_)
                    | StreamAccessKind::Sync => {
                        Some(Edge {
                            from: Node::Stream(*target_ref),
                            with: EdgeType::from(*kind),
                            to: Node::Stream(source_ref),
                        })
                    },
                }
            })
        })
    });

    let window_edges = mir.sliding_windows.iter().flat_map(|window| {
        [
            Edge {
                from: Node::Stream(window.caller),
                with: StreamAccessKind::SlidingWindow(window.reference).into(),
                to: Node::Window(window.reference),
            },
            Edge {
                from: Node::Window(window.reference),
                with: StreamAccessKind::SlidingWindow(window.reference).into(),
                to: Node::Stream(window.target),
            },
        ]
    });

    let trigger_edges = mir.triggers.iter().map(|trigger| {
        Edge {
            from: Node::Trigger(trigger.trigger_reference),
            with: StreamAccessKind::Sync.into(),
            to: Node::Stream(trigger.reference),
        }
    });

    access_edges.chain(window_edges).chain(trigger_edges).collect()
}

impl<'a> dot::Labeller<'a, Node, Edge> for DependencyGraph<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("example1").unwrap()
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
                value_ty,
                reference: _,
                expression: _,
            } => {
                format!(
                    "{stream_name}: {value_ty}<br/>Pacing: {pacing_ty}<br/>Memory Bound: {memory_bound}<br/>Layer {eval_layer}"
                )
            },
            NodeInformation::Window {
                reference,
                operation,
                duration,
            } => format!("Window {reference}<br/>Window Operation: {operation}<br/>Duration: {duration}"),
            NodeInformation::Trigger {
                idx,
                eval_layer,
                pacing_ty,
                value_ty,
                message,
                expression: _,
            } => {
                format!("Trigger {idx}: {value_ty}<br/>Pacing: {pacing_ty}<br/>Layer: {eval_layer}<br/><br/>{message}")
            },
        };

        dot::LabelText::HtmlStr(label_text.into())
    }

    fn edge_label<'b>(&'b self, edge: &Edge) -> LabelText<'b> {
        let kind = &edge.with.0;
        let label = match kind {
            StreamAccessKind::Sync => "Sync".into(),
            StreamAccessKind::Hold => "Hold".into(),
            StreamAccessKind::Offset(o) => format!("Offset({o})"),
            StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => "".into(),
            StreamAccessKind::Get => todo!(),
            StreamAccessKind::Fresh => todo!(),
        };
        dot::LabelText::LabelStr(label.into())
    }

    fn edge_style<'b>(&'b self, edge: &Edge) -> Style {
        let kind = &edge.with.0;
        match kind {
            StreamAccessKind::Hold => Style::Dashed,
            StreamAccessKind::Sync
            | StreamAccessKind::Offset(_)
            | StreamAccessKind::DiscreteWindow(_)
            | StreamAccessKind::SlidingWindow(_) => Style::None,
            StreamAccessKind::Get | StreamAccessKind::Fresh => todo!(),
        }
    }

    fn node_shape<'b>(&'b self, node: &Node) -> Option<LabelText<'b>> {
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
        Cow::Borrowed(&self.edges)
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

    macro_rules! build_sak {
        ( SW, $i:expr ) => {
            StreamAccessKind::SlidingWindow(WindowReference::Sliding($i))
        };
        ( DW, $i:expr ) => {
            StreamAccessKind::DiscreteWindow(WindowReference::Discrete($i))
        };
        ( $sak:ident ) => {
            StreamAccessKind::$sak
        };
    }

    // https://stackoverflow.com/a/34324856
    macro_rules! count {
        () => (0usize);
        ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
    }

    macro_rules! test_dependency_graph {
        ( $name:ident, $spec:literal, $( $edge_from_ty:ident($edge_from_i:expr) => $edge_to_ty:ident($edge_to_i:expr) : $with:ident $(($p:expr))? , )+ ) => {

            #[test]
            fn $name() {
                let config = ParserConfig::for_string($spec.into());
                let mir = parse(config).expect("should parse");
                let dep_graph = mir.dependency_graph();
                println!("{}", dep_graph.dot());
                let edges = &dep_graph.edges;
                $(
                    let from_node = build_node!($edge_from_ty($edge_from_i));
                    let to_node = build_node!($edge_to_ty($edge_to_i));
                    let with = build_sak!($with $(,$p)?);
                    let expected_edge = Edge {
                        from: from_node, to: to_node, with: with.into()
                    };
                    assert!(edges.iter().any(|edge| *edge == expected_edge), "specification did not contain expected edge {:#?}", expected_edge);
                )+
                assert!(edges.len() == count!($($with)+), "dependency graph had unwanted additional edges");
            }
        };
    }

    test_dependency_graph!(simple, "
        input a : UInt64
        input b : UInt64
        output c := a + b",
        Out(0) => In(0) : Sync,
        Out(0) => In(1) : Sync,
    );

    test_dependency_graph!(trigger, "
        input a : UInt64
        trigger a > 5",
        T(0) => Out(0) : Sync,
        Out(0) => In(0) : Sync,
    );

    test_dependency_graph!(more_complex, "
        input a : UInt64
        input b : UInt64
        output c := a + b.hold().defaults(to:0)
        output d@1Hz := a.aggregate(over:5s, using:count)
        // trigger c < 5
        trigger d < 5",
        Out(0) => In(0) : Sync,
        Out(0) => In(1) : Hold,
        Out(1) => SW(0) : SW(0),
        SW(0) => In(0) : SW(0),
        T(0) => Out(2) : Sync,
        Out(2) => Out(1) : Sync,
    );
}
